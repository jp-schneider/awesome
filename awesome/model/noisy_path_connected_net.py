import copy
import logging
import os
from typing import Dict, Optional, Tuple, Union, Any, List, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from awesome.dataset.prior_dataset import PriorManager
from awesome.model.convex_net import ConvexNet, ConvexNextNet, WNConvexNextNet
from awesome.model.diffeomorphism_net import DiffeomorphismNet, NormalizingFlow1D
from awesome.model.path_connected_net import PathConnectedNet
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.batcherize import batcherize
from awesome.util.pixelize import pixelize
from typing import Any, TYPE_CHECKING
import torch
from torch.utils.data.dataset import Dataset
from abc import abstractmethod
from awesome.util.temporary_property import TemporaryProperty
from tqdm.auto import tqdm
from awesome.util.torch import TensorUtil, get_weight_normalized_param_groups
import math
from awesome.model.zoo import Zoo

if TYPE_CHECKING:
    from awesome.agent.torch_agent import TorchAgent
else:
    TorchAgent = Any  # Avoiding import error
    PriorDataset = Any


    
class NoisyPathConnectedNet(PathConnectedNet):
    """Subclass of PathConnectedNet, for noisy spatio-temporal demonstration."""

    def _non_prior_based_pretrain(self,
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

        batch_size = kwargs.get("batch_size", 1)
        dataloader_shuffle = kwargs.get("dataloader_shuffle", False)

        criterion = kwargs.get("criterion", UnariesWeightedLoss(SE("mean")))

        prefit_flow_net_identity = kwargs.get("prefit_flow_net_identity", False)
        prefit_flow_net_identity_lr = kwargs.get("prefit_flow_net_identity_lr", 1e-2)
        prefit_flow_net_identity_weight_decay = kwargs.get("prefit_flow_net_identity_weight_decay", 1e-5)
        prefit_flow_net_identity_num_epochs = kwargs.get("prefit_flow_net_identity_num_epochs", 100)

        prefit_convex_net = kwargs.get("prefit_convex_net", False)
        prefit_convex_net_lr = kwargs.get("prefit_convex_net_lr", 1e-3)
        prefit_convex_net_weight_decay = kwargs.get("prefit_convex_net_weight_decay", 0)
        prefit_convex_net_num_epochs = kwargs.get("prefit_convex_net_num_epochs", 200)

        noisy_percentage = kwargs.get("noisy_percentage", 0.333)

        zoo = kwargs.get("zoo", None)
        
        logger: Tensorboard = agent.logger if use_logger else None
        batch_progress_bar = None
        training_state = wrapper_module.training

        if do_pretrain_checkpoints:
            if pretrain_checkpoint_dir is None:
                raise ValueError("Pretrain checkpoint dir must be provided.")
            if not os.path.exists(pretrain_checkpoint_dir):
                os.makedirs(pretrain_checkpoint_dir)

        noisy_unaries_dict = None

        try:
            with TemporaryProperty(wrapper_module, evaluate_prior=False), TemporaryProperty(agent.training_dataset, returns_index=True):
                # Deactivate prior evaluation for pretraining, as we will use the unaries separately
                wrapper_module.eval()

                if prefit_flow_net_identity or prefit_convex_net:     
                    inputs, _, _, _ = agent._decompose_training_item(train_set[0])
                    device_inputs: torch.Tensor = TensorUtil.to(
                        inputs, device=device)
                    # Getting inputs for prior
                    prior_args, prior_kwargs = wrapper_module.get_prior_args(device_inputs[0],
                                                                                *device_inputs[1:],
                                                                                segm=None,
                                                                                )

                    if prefit_flow_net_identity:
                        # Fit flow net identity
                        import torchvision.transforms.v2.functional as F

                        grid = self.create_normalized_grid((len(train_set), *prior_args[0].shape[-2:]))
                    
                        self.learn_flow_identity(grid, 
                                                lr=prefit_flow_net_identity_lr, 
                                                weight_decay=prefit_flow_net_identity_weight_decay,
                                                max_iter=prefit_flow_net_identity_num_epochs, 
                                                device=device,
                                                zoo=zoo,
                                                use_progress_bar=use_progress_bar,
                                                batch_size=batch_size
                                                )
                    if prefit_convex_net:
                        
                        inputs_2, _, _, _ = agent._decompose_training_item(train_set[-1])
                        device_inputs_2: torch.Tensor = TensorUtil.to(
                                                inputs_2, device=device)
                        # Stacking the 0 and the last unaries ontop to fit a convex net around the first and last frame.
                        #_inputs_comb = TensorUtil.apply_deep
                        _inputs_comb = [torch.stack((in_0, in_1), dim=0) for in_0, in_1 in zip(device_inputs, device_inputs_2) if isinstance(in_0, torch.Tensor)]

                        # Get the unaries
                        with torch.no_grad():
                            if isinstance(_inputs_comb, list):
                                unaries = wrapper_module(*_inputs_comb)
                            else:
                                unaries = wrapper_module(_inputs_comb)
                        
                        prior_args, _ = wrapper_module.get_prior_args(_inputs_comb[0],
                                                                                    *_inputs_comb[1:],
                                                                                    )
                        # Fit convex net
                        self.learn_convex_net(prior_args[0], 
                                            unaries,
                                            mode="unaries",
                                            use_deformed_grid=True,
                                            lr=prefit_convex_net_lr, 
                                            weight_decay=prefit_convex_net_weight_decay,
                                            max_iter=prefit_convex_net_num_epochs, 
                                            device=device,
                                            use_progress_bar=use_progress_bar
                                            )


                self.train()
            
                # Create optimizer
                groups = []
                groups.append(dict(params=self.flow_net.parameters(), weight_decay=flow_weight_decay))
                groups.append(dict(params=self.convex_net.parameters()))
                groups.append(dict(params=self.linear.parameters()))
                        
                optimizer = torch.optim.Adamax(groups, lr=lr)

                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=200, factor=0.5, verbose=True)

                # Pre-determine, which indices will be noisy => We ignore the first and last index, as we prefit the convex with them, technically this should not have an effect
                all_indices = torch.arange(1, len(train_set) - 1)
                noisy_indices = np.random.choice(all_indices.numpy(), size=int(round((len(train_set) * noisy_percentage))), replace=False).tolist()
                noisy_unaries_dict = {k: None for k in noisy_indices}


                # Train n iterations
                epoch_it = range(num_epochs)
                if use_progress_bar:
                    epoch_it = tqdm(epoch_it, desc="Pretraining epochs")

                for epoch in epoch_it:
                    # Iterate over the training dataset
                    step_it = DataLoader(train_set, batch_size=batch_size, shuffle=dataloader_shuffle)
                    if use_progress_bar:
                        if batch_progress_bar is None:
                            batch_progress_bar = tqdm(desc="Pretraining batches", total=len(step_it), delay=2)
                        else:
                            batch_progress_bar.reset(total=len(step_it))

                    epoch_loss = torch.tensor(0., device=device)
                    n_steps = len(step_it)

                    for step, item in enumerate(step_it):
                        inputs, labels, indices, prior_state = agent._decompose_training_item(
                            item)
                        device_inputs: torch.Tensor = TensorUtil.to(
                            inputs, device=device)
                        
                        unaries = None
                        # Get the unaries
                        with torch.no_grad():
                            if isinstance(device_inputs, list):
                                unaries = wrapper_module(*device_inputs)
                            else:
                                unaries = wrapper_module(device_inputs)

                        # Check of any of the indices is in the noisy indices
                        noisy_index = {k.item(): i for i, k in enumerate(indices) if k.item() in noisy_unaries_dict}
                        # Replace the unaries with the noisy ones
                        for noisy_idx, batch_idx in noisy_index.items():
                            # Get the noisy unaries
                            noisy_replacement = noisy_unaries_dict.get(noisy_idx, None)
                            if noisy_replacement is None:
                                noisy_replacement = torch.randn_like(unaries[batch_idx]) + 0.5
                                # Clamp to [0, 1]
                                noisy_replacement = torch.clamp(noisy_replacement, 0., 1.)
                                noisy_unaries_dict[noisy_idx] = noisy_replacement

                            # Replace the unaries
                            unaries[batch_idx] = noisy_replacement

                        # Getting inputs for prior
                        prior_args, prior_kwargs = wrapper_module.get_prior_args(device_inputs[0],
                                                                                    *device_inputs[1:],
                                                                                    segm=unaries[0, ...],
                                                                                    )
                        _input = prior_args[0]
                        device_prior_output = None

                        with torch.set_grad_enabled(True):
                            optimizer.zero_grad()
                            # Forward pass
                            device_prior_output = self(_input, *prior_args[1:], **prior_kwargs)
                            device_prior_output = wrapper_module.process_prior_output(
                                device_prior_output, use_sigmoid=use_prior_sigmoid, squeeze=False)

                            loss: torch.Tensor = criterion(
                                device_prior_output, unaries)

                            if ~(torch.isnan(loss) | torch.isinf(loss)):
                                loss.backward()
                                optimizer.step()
                            else:
                                raise ValueError("Loss is nan or inf!")
                            
                            self.enforce_convexity()

                            loss_item = loss.item()
                            epoch_loss += (1 / n_steps) * loss_item

                            if use_logger and use_step_logger:
                                logger.log_value(loss_item, f"PretrainingLoss/Batch", step=(n_steps * epoch) + step)

                            if batch_progress_bar is not None:
                                batch_progress_bar.set_postfix(
                                    loss=loss_item, refresh=False)
                                batch_progress_bar.update(1)
                    
                    if use_logger:
                        logger.log_value(epoch_loss, f"PretrainingLoss/Epoch", step=epoch)
                    lr_scheduler.step(epoch_loss, epoch=epoch)
            # The state if pretraining will be the state the model, as there are no priors per image
            state = self.state_dict()
        finally:
            # Save the noise unaries as well as the indices they where used for
            torch.save(noisy_unaries_dict, os.path.join(agent.agent_folder, "noisy_unaries_dict.pth"))        
            
            wrapper_module.train(training_state)
            self.train(training_state)
            if batch_progress_bar is not None:
                batch_progress_bar.close()
        return state
