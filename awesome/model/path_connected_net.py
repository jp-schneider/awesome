import logging
import os
from typing import Optional, Tuple, Any, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from awesome.dataset.prior_dataset import PriorManager
from awesome.model.convex_net import ConvexNet
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.batcherize import batcherize
from typing import Any, TYPE_CHECKING
import torch
from torch.utils.data.dataset import Dataset
from awesome.util.temporary_property import TemporaryProperty
from tqdm.auto import tqdm
from awesome.util.torch import TensorUtil
import math
from awesome.model.zoo import Zoo

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

def load_pretrain_checkpoint(model: torch.nn.Module, path: str, device: Optional[torch.device]) -> bool:
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
        return True
    except Exception as e:
        logging.error(f"Could not load pretrain checkpoint from {path}. Error: {e}")
        return False

def save_pretrain_checkpoint(model: torch.nn.Module, path: str) -> bool:
    try:
        torch.save(model.state_dict(), path)
        return True
    except Exception as e:
        logging.error(f"Could not save pretrain checkpoint to {path}. Error: {e}")
        return False
    
class PathConnectedNet(nn.Module, PretrainableModule):


    def __init__(self,
                 convex_net: ConvexNet,
                 flow_net: nn.Module,
                 in_channels: int = 2,
                 **kwargs):
        # call constructor from superclass
        super().__init__()
        self.convex_net = convex_net
        self.flow_net = flow_net
        self.linear = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels) # 1 By 1 convolution for translations
        self.linear.apply(self.weights_init_linear)

    def reset_parameters(self) -> None:
        self.convex_net.reset_parameters()
        self.flow_net.reset_parameters()
        self.linear.apply(self.weights_init_linear)

    def weights_init_linear(self, m):
        classname = type(m).__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    @batcherize()
    def forward(self, x):
        # Linear transformation of coordinates for easier global translation
        x = self.linear(x)
        xd = self.flow_net(x)
        xc = self.convex_net(xd)
        return xc

    def inverse_1b1_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of the 1 by 1 linear transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input.

        Returns
        -------
        torch.Tensor
            Inverse.
        """
        out = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for c in range(x.shape[1]):
                out[i, c] = (1 / self.linear.weight[c, 0]) * (x[i, c] - self.linear.bias[c])
        return out


    def inverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Inverse of the flow net part => Inverting the get_deformation.

        Parameters
        ----------
        x : torch.Tensor
            Output of get_deformation.

        Returns
        -------
        torch.Tensor
            Inverse
        """
        xd = self.flow_net.inverse(x)
        xl = self.inverse_1b1_linear(xd)
        return xl
        #return xc

    @batcherize()
    def get_deformation(self, x):
        x = self.linear(x)
        xd = self.flow_net(x)
        return xd


    def save_state(self, path: str) -> None:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.state_dict(), path)

    def load_state(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def enforce_convexity(self) -> None:
        self.convex_net.enforce_convexity()

    def get_unary_circle_approximation(self, unaries: torch.Tensor) -> torch.Tensor:
        # Very simple approximation of the circle by the center of mass and the area of the unaries
        area = unaries.sum()
        com = torch.argwhere(unaries.squeeze() > 0.).to(dtype=torch.float32).mean(dim=(0)).cpu()
        # Calculate the radius of the circle by the area
        radius = math.sqrt(area / math.pi)
        circle = self.create_circle(tuple(unaries.shape[-2:]), radius, com)
        if len(unaries.shape) == 3:
            circle = circle.unsqueeze(0)
        return circle

    def learn_flow_identity(self, 
                            x: torch.Tensor, 
                            lr: float = 1e-2,
                            weight_decay: float = 1e-5,
                            max_iter: int = 1000,
                            device: Optional[torch.device] = None,
                            zoo: Optional[Zoo] = None,
                            use_progress_bar: bool = True,
                            batch_size: int = 1
                            ) -> torch.Tensor:
        from awesome.measures.se import SE
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        #y = x.clone()
        criterion = SE()


        model = self.flow_net

        zoo_config = None
        zoo_name = "flow_identity"
        # Load pretrained model if available
        if zoo is not None:
            zoo_config = dict(
                lr=lr,
                weight_decay=weight_decay,
                max_iter=max_iter,
                criterion=criterion,
                x_data=TensorUtil.to_hash(x),
                batch_size=batch_size,
            )
            loaded, context = zoo.load_model_state(zoo_name, model, config=zoo_config)
            if loaded:
                logging.info("Loaded pretrained flow identity model!")
                return context.get("loss_hist", None)

        old_device = next(model.parameters()).device
        
        loss_hist = torch.zeros(max_iter, requires_grad=False)
        torch.fill(loss_hist, torch.nan)

        dataset = torch.utils.data.TensorDataset(x, x) # Identity mapping
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


        is_training = model.training
        if old_device != device:
            model.to(device)
            loss_hist = loss_hist.to(device)
        
        try:
            model.train()
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)

            it = range(max_iter)

            if use_progress_bar:
                it = tqdm(it, total=max_iter, desc="Learning flow identity", delay=5)


            for i in it:
                for x_in, y_true in dataloader:
                    x_in = x_in.to(device)
                    y_true = y_true.to(device)

                    optimizer.zero_grad()
                    y_pred = model(x_in)
                    loss = criterion(y_pred, y_true)

                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()
                    else:
                        raise ValueError("Loss is nan or inf!")
                    
                    loss_hist[i] = loss.detach()
                    if use_progress_bar:
                        it.set_postfix({"loss": "{:.3e}".format(loss.item())})

        finally:
            # Restore old device
            if old_device != device:
                model.to(old_device)
                loss_hist = loss_hist.to(old_device)

            model.train(is_training)
        
        # Save model state if zoo is available
        if zoo is not None:
            zoo.save_model_state(zoo_name, model, config=zoo_config, context=dict(loss_hist=loss_hist.cpu()))
        
        return loss_hist

    @classmethod
    def create_coordinate_grid(cls, grid_shape: Tuple[int, ...]) -> torch.Tensor:
        """Creates an n-dimensional coordinate grid.

        Parameters
        ----------
        grid_shape : Tuple[int, ...]
            Specifies the shape of the coordinate grid.

        Returns
        -------
        torch.Tensor
            The coordinate grid.
        """
        aranges = []
        for i in range(len(grid_shape)):
            aranges.append(torch.arange(grid_shape[i]).float())
        grid = torch.stack(torch.meshgrid(aranges)[::-1]) # To create (x, y[, z]) instead of ([z, ]y, x)
        if len(grid.shape) == 4:
            grid = grid.swapaxes(0, 1) # Swap to get z to batch dim
        return grid

    @classmethod
    def create_normalized_grid(cls, grid_shape: Tuple[int, ...]) -> torch.Tensor:
        """Creates a normalized n-dimensional coordinate grid.
        Grid values are in the range [0, 1].

        Parameters
        ----------
        grid_shape : Tuple[int, ...]
            The shape of the grid.

        Returns
        -------
        torch.Tensor
            A normalized coordinate grid in shape B x C x H x W.
        """        
        from awesome.transforms.min_max import MinMax
        grid = cls.create_coordinate_grid(grid_shape)
        dim = (0, 2, 3)
        if len(grid.shape) == 3:
            grid = grid.unsqueeze(0)
        min_max_norm = MinMax(0., 1., dim=dim)
        norm_grid = min_max_norm.fit_transform(grid)
        return norm_grid

    def create_circle(self, 
                            grid_shape: Tuple[int, ...], 
                            radius: float, 
                            center: Tuple[float, ...]) -> torch.Tensor:
        grid = PathConnectedNet.create_coordinate_grid(grid_shape)
        yy, xx = grid
        circle = ((yy - center[0])**2 + (xx - center[1])**2) <= radius**2
        return circle
    
    def learn_convex_net(self, 
                            x: torch.Tensor, 
                            unaries: torch.Tensor,
                            mode: Literal["circle", "unaries"] = "circle",
                            use_deformed_grid: bool = True,
                            lr: float = 1e-3,
                            weight_decay: float = 0,
                            max_iter: int = 1000,
                            device: Optional[torch.device] = None,
                            use_progress_bar: bool = True,
                            ) -> torch.Tensor:
        from awesome.measures.se import SE
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(unaries.shape) != 4:
            unaries = unaries.unsqueeze(0)

        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        if use_deformed_grid:
            with torch.no_grad():
                x = self.get_deformation(x)

        if mode == "circle":
            circle = self.get_unary_circle_approximation(1 - unaries[0])[None, ...]
            y = 1 - circle.float()
        elif mode == "unaries":
            y = unaries
        else:
            raise ValueError("Mode must be either 'circle' or 'unaries'!")

        criterion = SE()

        model = self.convex_net

        old_device = next(model.parameters()).device
        
        loss_hist = torch.zeros(max_iter)
        torch.fill(loss_hist, torch.nan)

        is_training = model.training
        if old_device != device:
            model.to(device)
            x = x.to(device)
            y = y.to(device)
            loss_hist = loss_hist.to(device)
        
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            it = range(max_iter) 
            if use_progress_bar:
                it = tqdm(it, total=max_iter, desc="Learning convex net", delay=5)
            
            for i in it:
                optimizer.zero_grad()
                y_pred = torch.sigmoid(model(x))
                loss = criterion(y_pred, y)

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    model.enforce_convexity()
                else:
                    raise ValueError("Loss is nan or inf!")
                
                loss_hist[i] = loss.detach()

                if use_progress_bar:
                    it.set_postfix({"loss": "{:.3e}".format(loss.item())})

        finally:
            # Restore old device
            if old_device != device:
                model.to(old_device)
                x = x.to(old_device)
                y = y.to(old_device)
                loss_hist = loss_hist.to(old_device)

            model.train(is_training)
        return loss_hist 

    def pretrain_unaries(self, 
                            x: torch.Tensor, 
                            unaries: torch.Tensor,
                            lr: float = 2e-3,
                            weight_decay: float = 1e-5,
                            max_iter: int = 1000,
                            device: Optional[torch.device] = None
                            ) -> None:
        from awesome.measures.se import SE

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(unaries.shape) != 4:
            unaries = unaries.unsqueeze(0)

        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        model = self

        groups = []
        groups.append(dict(params=model.flow_net.parameters(), weight_decay=weight_decay))
        groups.append(dict(params=model.convex_net.parameters()))
        groups.append(dict(params=model.linear.parameters()))

        optimizer = torch.optim.Adamax(groups, lr=lr)

        device = torch.device('cuda:0')

        criterion = SE()

        loss_hist = np.zeros(max_iter)
        loss_hist.fill(np.nan)

        output = None
        loss = None
        
        is_training = model.training
        model.train()

        old_device = next(model.parameters()).device

        if old_device != device:
            model.to(device)

        x = x.to(device)
        y = (1 - unaries).to(device)

        try:
            it = tqdm(range(max_iter), total=max_iter)

            for i in it:
                
                optimizer.zero_grad()
                output = torch.sigmoid(model(x))
                    
                loss = criterion(output, y)

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    self.enforce_convexity()
                else:
                    raise ValueError("Loss is nan or inf!")

                loss_numpy = loss.detach().cpu().numpy()
                loss_hist[i] = loss_numpy
                it.set_postfix({'loss': loss_numpy})

        finally:
            # Restore old device
            if old_device != device:
                model.to(old_device)
                x = x.to(old_device)
                y = y.to(old_device)
                
            model.train(is_training)
        return loss_hist

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
            # Spatio Temporality based pretraining
            return self._non_prior_based_pretrain(train_set=train_set,
                                                  test_set=test_set,
                                                device=device,
                                                agent=agent,
                                                use_progress_bar=use_progress_bar,
                                                do_pretrain_checkpoints=do_pretrain_checkpoints,
                                                use_pretrain_checkpoints=use_pretrain_checkpoints,
                                                pretrain_checkpoint_dir=pretrain_checkpoint_dir,
                                                wrapper_module=wrapper_module,
                                                **kwargs)

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
            with TemporaryProperty(wrapper_module, evaluate_prior=False):
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
            wrapper_module.train(training_state)
            self.train(training_state)
            if batch_progress_bar is not None:
                batch_progress_bar.close()
        return state

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
            data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
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
                            success = load_pretrain_checkpoint(self, ckp_path, device=device)
                            if success:
                                logging.info(f"Loaded pretrain checkpoint from {ckp_path}. Continuing with next image.")
                                has_proper_prior_fit = True
                                loaded_current_from_checkpoint = True


                    if not has_proper_prior_fit and (reuse_state and previous_state is not None):
                        load_state = TensorUtil.apply_deep(
                            previous_state, fnc=lambda x: x.to(device=device))
                        self.load_state_dict(load_state)
                    elif not loaded_current_from_checkpoint:
                        # If not reusing state, we will check if we fit flow net and convex net independently beforehand to get hopefully better results
                        if prefit_flow_net_identity:
                            # Fit flow net identity
                            self.learn_flow_identity(_input, 
                                                     lr=prefit_flow_net_identity_lr, 
                                                     weight_decay=prefit_flow_net_identity_weight_decay,
                                                     max_iter=prefit_flow_net_identity_num_epochs, 
                                                     device=device,
                                                     zoo=zoo,
                                                     use_progress_bar=use_progress_bar
                                                     )
                        if prefit_convex_net:
                            # Fit convex net
                            self.learn_convex_net(_input, 
                                                     unaries,
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
                        groups.append(dict(params=self.flow_net.parameters(), weight_decay=flow_weight_decay))
                        groups.append(dict(params=self.convex_net.parameters()))
                        groups.append(dict(params=self.linear.parameters()))
                                
                        optimizer = torch.optim.Adamax(groups, lr=lr)


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
                                self.enforce_convexity()
                                lr_scheduler.step(loss)

                                if use_logger and use_step_logger:
                                    logger.log_value(
                                        loss.item(), f"PretrainingLoss/Image_{i}", step=step)

                                if batch_progress_bar is not None:
                                    batch_progress_bar.set_postfix(
                                        loss=loss.item(), refresh=False)
                                    batch_progress_bar.update()
                        
                        if device_prior_output is not None:
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
                                        logging.info(f"Prior fit not proper on image index: {i}. Retrying. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                                        # Reset parameters and try again => Need more testing if this is a good idea
                                        self.reset_parameters()
                                    else:
                                        # We did not fit properly, but we retried enough times, continue with next image
                                        logging.info(f"Prior fit not proper on image index: {i}. Retries exceeded. Metric: {proper_fit_result} Threshold: {proper_prior_fit_threshold}")
                                num_retrys += 1
                        else:
                            logging.warning(f"Prior output is None => No training image index: {i}. Considering as fitted.")
                            has_proper_prior_fit = True

                    if reuse_state and has_proper_prior_fit:
                        current_state = self.state_dict()

                        def _copy(x):
                            return x.detach().clone()
                        # Detaches the state and copies it, so it can be reused.
                        previous_state = TensorUtil.apply_deep(
                            current_state, fnc=_copy)
                            
                    if do_pretrain_checkpoints and not loaded_current_from_checkpoint:
                        ckp_path = os.path.join(pretrain_checkpoint_dir, f"pretrain_checkpoint_{i}.pth")
                        save_pretrain_checkpoint(self, ckp_path)
        
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
