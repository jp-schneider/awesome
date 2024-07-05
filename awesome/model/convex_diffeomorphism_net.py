import copy
import logging
import os
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from awesome.dataset.prior_dataset import PriorManager
from awesome.model.convex_net import ConvexNet, ConvexNextNet, WNConvexNextNet
from awesome.model.diffeomorphism_net import DiffeomorphismNet, NormalizingFlow1D
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.pixelize import pixelize
from typing import Any, TYPE_CHECKING
import torch
from torch.utils.data.dataset import Dataset
from abc import abstractmethod
from awesome.util.temporary_property import TemporaryProperty
from tqdm.autonotebook import tqdm
from awesome.util.torch import TensorUtil, get_weight_normalized_param_groups
import math
from awesome.util.batcherize import batcherize

if TYPE_CHECKING:
    from awesome.agent.torch_agent import TorchAgent
else:
    TorchAgent = Any  # Avoiding import error
    PriorDataset = Any


def minmax(v: torch.Tensor,
           v_min: torch.Tensor,
           v_max: torch.Tensor,
           new_min: torch.Tensor = 0.,
           new_max: torch.Tensor = 1.,
           ):
    return (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


class ConvexDiffeomorphismNet(nn.Module, PretrainableModule):

    def translate_only_point(self, from_point: torch.Tensor, to_point: torch.Tensor, grid: torch.Tensor):
        """Translates / Shifts the output of the diffeomorphism net in such a way
        that to_point is mapped to the same point as from_point, resulting
        in a affine transformation of the prior. For this function only one point is required,
        the output will be just shifted, not rotated or scaled / sheared.

        Parameters
        ----------
        from_point : torch.Tensor
            Source point.
        to_point : torch.Tensor
            Target point.
        """
        # Create 2 additional points that all ar linearly independent
        # This is needed to compute the affine transformation
        res_from = torch.zeros((self.in_features + 1, self.in_features),
                               device=from_point.device, dtype=from_point.dtype)
        res_to = torch.zeros((self.in_features + 1, self.in_features),
                             device=to_point.device, dtype=to_point.dtype)

        res_from[0, :] = from_point
        res_to[0, :] = to_point

        add_value = 3
        for i in range(self.in_features):
            v = torch.zeros(self.in_features,
                            device=from_point.device, dtype=from_point.dtype)
            v[i] += add_value
            res_from[i+1, :] = res_from[0, :] + v
            res_to[i+1, :] = res_to[0, :] + v

        grid = grid.squeeze()
        res_from = grid[..., res_from[:, 1].to(
            dtype=torch.long), res_from[:, 0].to(dtype=torch.long)].T
        res_to = grid[..., res_to[:, 1].to(
            dtype=torch.long), res_to[:, 0].to(dtype=torch.long)].T

        self.translate(res_from, res_to)

    def translate(self, from_points: torch.Tensor, to_points: torch.Tensor):
        """Translates / Shifts the output of the diffeomorphism net in such a way
        that to_points are mapped to the same points as from_points, resulting 
        in a affine transformation of the prior. The points should be linar independent,
        otherwise the affine transformation is not unique and pytorch chooses any.

        Points are expected in the formant (B, C) where C is the value at x,y coordinate.

        Parameters
        ----------
        from_points : torch.Tensor
            Points which should be taken as reference.
        to_points : torch.Tensor
            To points which should be mapped to the from points.
        """
        # Sample number of output points from the first linear layer
        if from_points.shape != to_points.shape:
            raise ValueError("From and to points must have the same shape.")

        linear_layer = self.linear
        needed_features_points = int(math.ceil((linear_layer.out_features)))

        # Converting dtype and device
        to_points = to_points.to(
            dtype=linear_layer.weight.dtype, device=linear_layer.weight.device)
        from_points = from_points.to(
            dtype=linear_layer.weight.dtype, device=linear_layer.weight.device)

        if from_points.shape[0] < needed_features_points:
            # Sample more points
            raise ValueError("Not enough points to sample from. Need at least {} points, got {}.".format(
                needed_features_points, from_points.shape[0]))

        from_transf = None
        with torch.no_grad():
            from_transf = linear_layer(from_points)

        # Recomputing the weights and bias by enforcing the output of the linear layer to be equal for the from and to points
        X = torch.cat((to_points, torch.ones(
            (to_points.shape[0], 1), device=to_points.device, dtype=to_points.dtype)), dim=1)
        theta = torch.linalg.inv(X.T @ X) @ (X.T @ from_transf)

        weights = theta[:-1, :].T
        bias = theta[-1, :]

        linear_layer.weight.data = weights
        linear_layer.bias.data = bias

    def __init__(self,
                 n_hidden: int = 130,
                 n_hidden_layers: int = 1,
                 nf_layers: int = 4,
                 nf_hidden: int = 70,
                 in_features: int = 2,
                 diffeo_args: Dict[str, Any] = None,
                 **kwargs):
        # call constructor from superclass
        super().__init__()

        self.convex_net = ConvexNextNet(
            n_hidden=n_hidden,
            in_features=in_features,
            n_hidden_layers=n_hidden_layers)
        if diffeo_args is None:
            diffeo_args = dict()
        self.in_features = in_features
        # self.diffeo_net = DiffeomorphismNet()
        if "num_coupling" not in diffeo_args:
            diffeo_args["num_coupling"] = nf_layers
        if "width" not in diffeo_args:
            diffeo_args["width"] = nf_hidden
        if "in_features" not in diffeo_args:
            diffeo_args["in_features"] = in_features

        self.diffeo_net = NormalizingFlow1D(**diffeo_args)
        self.linear = nn.Linear(in_features, in_features)
        self.linear.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        classname = type(m).__name__
        if classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))
            m.bias.data.fill_(0)

    def reset_parameters(self) -> None:
        self.convex_net.reset_parameters()
        self.diffeo_net.reset_parameters()
        self.linear.apply(self.weights_init_normal)
        return True

    @pixelize()
    def forward(self, x):
        x = self.linear(x)
        xd = self.diffeo_net(x)
        xc = self.convex_net(xd)
        return xc

    @batcherize()
    @pixelize()
    def get_deformation(self, x):
        x = self.linear(x)
        xd = self.diffeo_net(x)
        return xd

    def enforce_convexity(self) -> None:
        self.convex_net.enforce_convexity()

    def pretrain(self,
                 train_set: Dataset,
                 test_set: Dataset,
                 device: torch.device,
                 agent: TorchAgent,
                 use_progress_bar: bool = True,
                 do_pretrain_checkpoints: bool = False,
                 use_pretrain_checkpoints: bool = False,
                 pretrain_checkpoint_dir: Optional[str] = None, 
                 wrapper_module: Optional[nn.Module] = None,
                 **kwargs
                 ) -> Any:
        from awesome.dataset.prior_dataset import PriorDataset
        from awesome.measures.unaries_weighted_loss import UnariesWeightedLoss
        from awesome.util.tensorboard import Tensorboard
        from awesome.measures.miou import MIOU

        if wrapper_module is None:
            raise ValueError("Wrapper model must be provided for pretraining.")
        if not isinstance(agent.training_dataset, PriorDataset):
            raise ValueError("Agent must be trained on a prior dataset.")

        state = None
        # Pretrain args

        num_epochs = kwargs.get("num_epochs", 2000)
        lr = kwargs.get("lr", 0.003)
        criterion = kwargs.get("criterion", UnariesWeightedLoss(nn.BCELoss(), mode="none"))
        use_prior_sigmoid = kwargs.get("use_prior_sigmoid", True)
        use_logger = kwargs.get("use_logger", False)
        use_step_logger = kwargs.get("use_step_logger", False)
        reuse_state = kwargs.get("reuse_state", True)
        reuse_state_epochs = kwargs.get("reuse_state_epochs", 200)


        proper_prior_fit_threshold = kwargs.get("proper_prior_fit_threshold", 0.5)
        proper_prior_fit_retrys = kwargs.get("proper_prior_fit_retrys", 1)
        proper_prior_fit_metric = MIOU(average="binary", 
                                                   invert=True # Invert to calculate agains fg 
                                                   )
        
        weight_decay_on_weight_g = kwargs.get("weight_decay_on_weight_g", 5e-5)
        weight_decay_on_convex_weight = kwargs.get("weight_decay_on_convex_weight", False)
        
        logger: Tensorboard = agent.logger if use_logger else None

        batch_progress_bar = None

        training_state = wrapper_module.training

        if do_pretrain_checkpoints:
            if pretrain_checkpoint_dir is None:
                raise ValueError("Pretrain checkpoint dir must be provided.")
            if not os.path.exists(pretrain_checkpoint_dir):
                os.makedirs(pretrain_checkpoint_dir)

        def _load_pretrain_checkpoint(path: str) -> bool:
            try:
                checkpoint = torch.load(path, map_location=device)
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                logging.error(f"Could not load pretrain checkpoint from {path}. Error: {e}")
                return False

        def _save_pretrain_checkpoint(path: str) -> bool:
            try:
                torch.save(self.state_dict(), path)
                return True
            except Exception as e:
                logging.error(f"Could not save pretrain checkpoint to {path}. Error: {e}")
                return False

        def center_of_mass(m: torch.Tensor) -> torch.Tensor:
            foreground = (1 - (m > 0.5).float()).bool().squeeze()
            center_of_mass = torch.sum(torch.argwhere(
                foreground), dim=0) / torch.sum(foreground).to(dtype=torch.long)
            return center_of_mass

        try:
            wrapper_module.eval()
            # Iterate over the training dataset
            data_loader = DataLoader(train_set, batch_size=1, shuffle=False)
            it = data_loader
            if use_progress_bar:
                it = tqdm(it, desc="Pretraining images")
                
            self.train()
            previous_state = None
            previous_center_of_mass = None

            for i, item in enumerate(it):
                inputs, labels, indices, prior_state = agent._decompose_training_item(
                    item)
                device_inputs: torch.Tensor = TensorUtil.to(
                    inputs, device=device)
                # device_labels: torch.Tensor = TensorUtil.to(labels, device=device)

                # Evaluate model to get unaries
                # Switch prior weights if needed, using context manager
                with PriorManager(wrapper_module,
                                  prior_state=prior_state,
                                  prior_cache=agent.training_dataset.__prior_cache__,
                                  model_device=device,
                                  training=True
                                  ):
                    
                    unaries = None
                    has_proper_prior_fit = False
                    loaded_current_from_checkpoint = False
        
                    # Get the unaries
                    # Disable prior evaluation to just get the unaries
                    with torch.no_grad(), TemporaryProperty(wrapper_module, evaluate_prior=False):
                        if isinstance(device_inputs, list):
                            unaries = wrapper_module(*device_inputs)
                        else:
                            unaries = wrapper_module(device_inputs)


                    # Getting inputs for prior
                    prior_args, prior_kwargs = wrapper_module.get_prior_args(device_inputs[0],
                                                                             *device_inputs[1:],
                                                                             segm=unaries[0, ...],
                                                                             )
                    _input = prior_args[0]
                    actual_input = _input.detach().clone()

                    _unique_vals = torch.unique(unaries >= 0.5)
                    # Check if unaries output contains at least some foreground
                    if len(_unique_vals) == 1:
                        # No foreground / background predicted. Skip this image
                        # We will keep the state of the prior if reuse_state is True
                        # If there was a pre existing state, we will use it again
                        logging.warning(f"Unaries of segmentation model contain no foreground. Skipping image. {i}")
                        continue

                    if use_pretrain_checkpoints:
                        ckp_path = os.path.join(pretrain_checkpoint_dir, f"pretrain_checkpoint_{i}.pth")
                        if os.path.exists(ckp_path):
                            success = _load_pretrain_checkpoint(ckp_path)
                            if success:
                                logging.info(f"Loaded pretrain checkpoint from {ckp_path}. Continuing with next image.")
                                has_proper_prior_fit = True
                                loaded_current_from_checkpoint = True


                    if not has_proper_prior_fit and (reuse_state and previous_state is not None):
                        load_state = TensorUtil.apply_deep(
                            previous_state, fnc=lambda x: x.to(device=device))
                        self.load_state_dict(load_state)
                        if len(_unique_vals) > 1:
                            # Shift the prior to the center of mass if there is foreground
                            com = center_of_mass(unaries).to(dtype=torch.long)
                            # Translate the prior to the center of mass
                            if previous_center_of_mass is not None:
                                self.translate_only_point(previous_center_of_mass.flip(
                                    dims=(-1,)), com.flip(dims=(-1,)), grid=_input.squeeze())
                            previous_center_of_mass = com

                    num_retrys = 0
                    proper_fit_result = -1

                    while not has_proper_prior_fit and num_retrys <= proper_prior_fit_retrys:
                        # Determine number of epochs
                        epochs = 0
                        if reuse_state and previous_state is not None and num_retrys == 0:
                            # If we reuse the state, we will only train for a few epochs, to not destroy the state / shape
                            # If we num_retrys > 0, we will train for the full number of epochs, as we already tried to fit the prior and failed
                            epochs = reuse_state_epochs
                        else:
                            # Full training
                            epochs = num_epochs
                        
                        # Train n iterations
                        it = range(epochs)
                        if use_progress_bar:
                            desc = f'Image {i + 1}: Pretraining'
                            if batch_progress_bar is None:
                                batch_progress_bar = tqdm(
                                    total=epochs,
                                    desc=desc,
                                    leave=True)
                            else:
                                batch_progress_bar.reset(total=epochs)
                                batch_progress_bar.set_description(desc)

                        # Create optimizer
                        groups = get_weight_normalized_param_groups(
                            self, weight_decay_on_weight_g, norm_suffix='weight_g')
                        
                        if weight_decay_on_convex_weight:
                            unnorm_group = next((x for x in groups if x['name'] == "unnormalized" ), None)
                            norm_group = next((x for x in groups if x['name'] == "normalized" ), None)
                            
                            unnorm_params = {id(x): x for x in unnorm_group['params']}
                            norm_params = {id(x): x for x in norm_group['params']}

                            for name, param in self.named_parameters():
                                if name.startswith("convex_net") and name.endswith("weight"):
                                    # Check if in unnormalized group
                                    if id(param) in unnorm_params:
                                        # Remove from unnormalized group
                                        unnorm_params.pop(id(param))
                                        # Add to normalized group
                                        norm_params[id(param)] = param
                            unnorm_group['params'] = list(unnorm_params.values())
                            norm_group['params'] = list(norm_params.values())
                                
                        optimizer = torch.optim.Adam(groups, lr=lr)
                        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, patience=200, factor=0.5, verbose=True)

                        device_prior_output = None

                        with torch.set_grad_enabled(True):
                            # Train n iterations
                            for step in it:
                                optimizer.zero_grad()
                                # Forward pass
                                device_prior_output = self(
                                    actual_input, *prior_args[1:], **prior_kwargs)
                                device_prior_output = wrapper_module.process_prior_output(
                                    device_prior_output, use_sigmoid=use_prior_sigmoid)[None, ...]  # Add batch dim again

                                loss: torch.Tensor = criterion(
                                    device_prior_output, unaries)

                                loss.backward()
                                optimizer.step()
                                lr_scheduler.step(loss)

                                if use_logger and use_step_logger:
                                    logger.log_value(
                                        loss.item(), f"PretrainingLoss/Image_{i}", step=step)

                                self.enforce_convexity()
                                if batch_progress_bar is not None:
                                    batch_progress_bar.set_postfix(
                                        loss=loss.item(), refresh=False)
                                    batch_progress_bar.update()
                        
                        with torch.no_grad():
                            # Check if the prior is fitted properly => Having at least proper_prior_fit_threshold foreground
                            proper_fit_result = proper_prior_fit_metric((device_prior_output > 0.5).float(), (unaries > 0.5).float()).detach().cpu().item()
                            if use_logger:
                                logger.log_value(proper_fit_result, f"PretrainingLoss/IoU", step=i)
                            if proper_fit_result >= proper_prior_fit_threshold:
                                has_proper_prior_fit = True
                                logging.info(f"Proper prior fit on image index: {i} got metric: {proper_fit_result}.")
                            else:
                                has_proper_prior_fit = False
                                if num_retrys < proper_prior_fit_retrys:
                                    logging.info(f"Prior fit not proper. Retrying. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                                    # Reset parameters and try again :D
                                    self.reset_parameters()
                                else:
                                    # We did not fit properly, but we retried enough times, continue with next image
                                    logging.info(f"Prior fit not proper. Retries exceeded. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                            num_retrys += 1

                    if reuse_state and has_proper_prior_fit:
                        current_state = self.state_dict()

                        def _copy(x):
                            return x.detach().clone()
                        # Detaches the state and copies it, so it can be reused.
                        previous_state = TensorUtil.apply_deep(
                            current_state, fnc=_copy)
                        if previous_center_of_mass is None:
                            # Compute center of mass
                            previous_center_of_mass = center_of_mass(
                                unaries).to(dtype=torch.long)
                            
                    if do_pretrain_checkpoints and not loaded_current_from_checkpoint:
                        ckp_path = os.path.join(pretrain_checkpoint_dir, f"pretrain_checkpoint_{i}.pth")
                        _save_pretrain_checkpoint(ckp_path)
        
            # The state if pretraining will be the state of all priors, which is managed by the prior cache
            state = agent.training_dataset.__prior_cache__.get_state()
        finally:
            wrapper_module.train(training_state)
            self.train(training_state)
            if batch_progress_bar is not None:
                batch_progress_bar.close()
        return state

    def pretrain_load_state(self,
                            train_set: Dataset,
                            test_set: Dataset,
                            device: torch.device,
                            agent: TorchAgent,
                            state: Any,
                            use_progress_bar: bool = True,
                            wrapper_module: Optional[nn.Module] = None,
                            **kwargs):
        agent.training_dataset.__prior_cache__.set_state(state)
