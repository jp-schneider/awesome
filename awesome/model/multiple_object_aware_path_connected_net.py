from awesome.model.abstract_combined_segmentation_module import EvaluationMode, GradientMode
from awesome.model.number_based_multi_prior_module import NumberBasedMultiPriorModule
from awesome.model.path_connected_net import PathConnectedNet, load_pretrain_checkpoint, save_pretrain_checkpoint, minmax
import logging
import os
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from awesome.dataset.prior_dataset import PriorManager, create_prior_collate_fn
from typing import Any, TYPE_CHECKING
import torch
from torch.utils.data.dataset import Dataset
from awesome.util.temporary_property import TemporaryProperty
from tqdm.auto import tqdm
from awesome.util.torch import TensorUtil

if TYPE_CHECKING:
    from awesome.agent.torch_agent import TorchAgent
else:
    TorchAgent = Any  # Avoiding import error
    PriorDataset = Any

class MultipleObjectsAwarePathConnectedNet(NumberBasedMultiPriorModule):


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
        # Check which pretraining must be performed => Prior based or not
        from awesome.dataset.prior_dataset import PriorDataset
        if isinstance(agent.training_dataset, PriorDataset) and agent.training_dataset.has_prior:
            # Prior based pretraining (per image)
            return self._prior_based_pretrain(train_set=train_set,
                                                test_set=test_set,
                                                device=device,
                                                agent=agent,
                                                use_progress_bar=use_progress_bar,
                                                do_pretrain_checkpoints=do_pretrain_checkpoints,
                                                use_pretrain_checkpoints=use_pretrain_checkpoints,
                                                pretrain_checkpoint_dir=pretrain_checkpoint_dir,
                                                wrapper_module=wrapper_module,
                                                **kwargs)
        else:
            raise NotImplementedError("Fixed pretraining not implemented.")
            
    
    def _prior_based_pretrain(self,
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
        from awesome.measures.se import SE
        from awesome.util.tensorboard import Tensorboard
        from awesome.measures.miou import MIOU

        if wrapper_module is None:
            raise ValueError("Wrapper model must be provided for pretraining.")
        if not isinstance(agent.training_dataset, PriorDataset):
            raise ValueError("Agent must be trained on a prior dataset.")

        state = None
        # Pretrain args

        num_epochs = kwargs.get("num_epochs", 2000)
        lr = kwargs.get("lr", 0.001)
        flow_weight_decay = kwargs.get("flow_weight_decay", 1e-5)


        use_prior_sigmoid = kwargs.get("use_prior_sigmoid", True)
        use_logger = kwargs.get("use_logger", False)
        use_step_logger = kwargs.get("use_step_logger", False)

        reuse_state = kwargs.get("reuse_state", True)
        reuse_state_epochs = kwargs.get("reuse_state_epochs", 200)
        batch_size = kwargs.get("batch_size", 1)
        criterion = kwargs.get("criterion", UnariesWeightedLoss(SE("mean")))


        prefit_flow_net_identity = kwargs.get("prefit_flow_net_identity", False)
        prefit_flow_net_identity_lr = kwargs.get("prefit_flow_net_identity_lr", 1e-2)
        prefit_flow_net_identity_weight_decay = kwargs.get("prefit_flow_net_identity_weight_decay", 1e-5)
        prefit_flow_net_identity_num_epochs = kwargs.get("prefit_flow_net_identity_num_epochs", 100)

        prefit_convex_net = kwargs.get("prefit_convex_net", False)
        prefit_convex_net_lr = kwargs.get("prefit_convex_net_lr", 1e-3)
        prefit_convex_net_weight_decay = kwargs.get("prefit_convex_net_weight_decay", 0)
        prefit_convex_net_num_epochs = kwargs.get("prefit_convex_net_num_epochs", 200)

        proper_prior_fit_threshold = kwargs.get("proper_prior_fit_threshold", 0.5)
        proper_prior_fit_retrys = kwargs.get("proper_prior_fit_retrys", 1)
        proper_prior_fit_metric = MIOU(average="binary", 
                                                   invert=True # Invert to calculate agains fg 
                                                   )
        zoo = kwargs.get("zoo", None)
        
        logger: Tensorboard = agent.logger if use_logger else None
        batch_progress_bar = None
        training_state = wrapper_module.training

        if do_pretrain_checkpoints:
            if pretrain_checkpoint_dir is None:
                raise ValueError("Pretrain checkpoint dir must be provided.")
            if not os.path.exists(pretrain_checkpoint_dir):
                os.makedirs(pretrain_checkpoint_dir)

        try:
            wrapper_module.eval()
            # Iterate over the training dataset

            collate_fn = None
            if self.use_prior_collate_fn:
                use_prior = isinstance(agent.training_dataset, PriorDataset) and agent.training_dataset.has_prior
                collate_fn = create_prior_collate_fn(has_prior=use_prior)

            data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
            it = data_loader
            if use_progress_bar:
                it = tqdm(it, desc="Pretraining images")
                
            self.train()

            previous_image_object_states = dict()
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
                    with torch.no_grad(), TemporaryProperty(wrapper_module, 
                                                            evaluation_mode=EvaluationMode.SEGMENTATION,
                                                            gradient_mode=GradientMode.NONE,
                                                            ):
                        if isinstance(device_inputs, list):
                            unaries = wrapper_module(*device_inputs)
                        else:
                            unaries = wrapper_module(device_inputs)

                    # Determine the number of channels to create corresponding amount of priors
                    n_priors = 1
                    if len(unaries.shape) == 4:
                        if unaries.shape[1] > 1:        
                            # First channel is the background
                            n_priors = unaries.shape[1] - 1
                    self.assure_prior_count(n_priors)

                    # Getting inputs for prior
                    prior_args, prior_kwargs = wrapper_module.get_prior_args(*device_inputs)
                    
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

                    previous_state = None
                    # Iterate over the objects within the image
                    for object_idx in range(n_priors): 
                        
                        # Only keep the last images state
                        for key in previous_image_object_states.keys():
                            if key < i - 1:
                                del previous_image_object_states[key]

                        object_unaries = unaries[:, object_idx, ...][:, None, ...] if n_priors > 1 else unaries
                        prior_model = self.priors[object_idx]
                        previous_state = previous_image_object_states.get(i - 1, dict()).get(object_idx, None)

                        if use_pretrain_checkpoints:
                            ckp_path = os.path.join(pretrain_checkpoint_dir, f"pretrain_checkpoint_{i}_{object_idx}.pth")
                            if os.path.exists(ckp_path):
                                success = load_pretrain_checkpoint(prior_model, ckp_path, device=device)
                                if success:
                                    logging.info(f"Loaded pretrain checkpoint from {ckp_path}. Continuing with next image / object.")
                                    has_proper_prior_fit = True
                                    loaded_current_from_checkpoint = True

                        if not has_proper_prior_fit and (reuse_state and previous_state is not None):
                            load_state = TensorUtil.apply_deep(
                                previous_state, fnc=lambda x: x.to(device=device))
                            prior_model.load_state_dict(load_state)

                        elif not loaded_current_from_checkpoint:
                            # If not reusing state, we will check if we fit flow net and convex net independently beforehand to get hopefully better results
                            if prefit_flow_net_identity:
                                # Fit flow net identity
                                prior_model.learn_flow_identity(_input, 
                                                        lr=prefit_flow_net_identity_lr, 
                                                        weight_decay=prefit_flow_net_identity_weight_decay,
                                                        max_iter=prefit_flow_net_identity_num_epochs, 
                                                        device=device,
                                                        zoo=zoo,
                                                        use_progress_bar=use_progress_bar
                                                        )
                            if prefit_convex_net:
                                # Fit convex net
                                prior_model.learn_convex_net(_input, 
                                                        object_unaries,
                                                        mode="unaries",
                                                        use_deformed_grid=True,
                                                        lr=prefit_convex_net_lr, 
                                                        weight_decay=prefit_convex_net_weight_decay,
                                                        max_iter=prefit_convex_net_num_epochs, 
                                                        device=device,
                                                        use_progress_bar=use_progress_bar
                                                        )

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
                            groups = []
                            groups.append(dict(params=prior_model.flow_net.parameters(), weight_decay=flow_weight_decay))
                            groups.append(dict(params=prior_model.convex_net.parameters()))
                            groups.append(dict(params=prior_model.linear.parameters()))
                                    
                            optimizer = torch.optim.Adamax(groups, lr=lr)


                            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, patience=200, factor=0.5, verbose=True)

                            device_prior_output = None

                            with torch.set_grad_enabled(True):
                                # Train n iterations
                                for step in it:
                                    optimizer.zero_grad()
                                    # Forward pass
                                    device_prior_output = prior_model(
                                        actual_input, *prior_args[1:], **prior_kwargs)
                                    device_prior_output = wrapper_module.process_prior_output(
                                        device_prior_output)

                                    loss: torch.Tensor = criterion(
                                        device_prior_output, unaries)

                                    loss.backward()
                                    optimizer.step()
                                    prior_model.enforce_convexity()
                                    lr_scheduler.step(loss)

                                    if use_logger and use_step_logger:
                                        logger.log_value(
                                            loss.item(), f"PretrainingLoss/Image_{i}_{object_idx}", step=step)

                                    if batch_progress_bar is not None:
                                        batch_progress_bar.set_postfix(
                                            loss=loss.item(), refresh=False)
                                        batch_progress_bar.update()
                            
                            if device_prior_output is not None:
                                with torch.no_grad():
                                    # Check if the prior is fitted properly => Having at least proper_prior_fit_threshold foreground
                                    proper_fit_result = proper_prior_fit_metric((device_prior_output > 0.5).float(), (object_unaries > 0.5).float()).detach().cpu().item()
                                    if use_logger:
                                        logger.log_value(proper_fit_result, f"PretrainingLoss/IoU", step=i)
                                    if proper_fit_result >= proper_prior_fit_threshold:
                                        has_proper_prior_fit = True
                                        logging.info(f"Proper prior fit on image index: {i} got metric: {proper_fit_result}.")
                                    else:
                                        has_proper_prior_fit = False
                                        if num_retrys < proper_prior_fit_retrys:
                                            logging.info(f"Prior fit not proper on image index: {i}. Retrying. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                                            # Reset parameters and try again => Need more testing if this is a good idea
                                            prior_model.reset_parameters()
                                        else:
                                            # We did not fit properly, but we retried enough times, continue with next image
                                            logging.info(f"Prior fit not proper on image index: {i}. Retries exceeded. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                                    num_retrys += 1
                            else:
                                logging.warning(f"Prior output is None => No training image index: {i}. Considering as fitted.")
                                has_proper_prior_fit = True

                        if reuse_state and has_proper_prior_fit:
                            current_state = prior_model.state_dict()

                            def _copy(x):
                                return x.detach().clone()
                            # Detaches the state and copies it, so it can be reused.
                            detached_state = TensorUtil.apply_deep(
                                current_state, fnc=_copy)
                            
                            states = previous_image_object_states.get(i, dict())
                            states[object_idx] = detached_state
                            previous_image_object_states[i] = states
                                
                        if do_pretrain_checkpoints and not loaded_current_from_checkpoint:
                            ckp_path = os.path.join(pretrain_checkpoint_dir, f"pretrain_checkpoint_{i}_{object_idx}.pth")
                            save_pretrain_checkpoint(self, ckp_path)

                        
        
            # The state if pretraining will be the state of all priors, which is managed by the prior cache
            state = agent.training_dataset.__prior_cache__.get_state()
        finally:
            wrapper_module.train(training_state)
            self.train(training_state)
            if batch_progress_bar is not None:
                batch_progress_bar.close()
        return state