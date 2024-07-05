import logging
import torch
from awesome.model.multiple_object_aware_path_connected_net import MultipleObjectsAwarePathConnectedNet
from awesome.transforms.transform import Transform

from typing import Any, Dict, Literal, Optional, Type
from awesome.model.path_connected_net import PathConnectedNet
from awesome.model.convex_net import ConvexNextNet
from awesome.transforms.mean_std import MeanStd
from awesome.transforms.min_max import MinMax

import normflows as nf
from awesome.model.norm_net import NormNet
from awesome.model.path_connected_net import PathConnectedNet
from awesome.model.pixelize_net import PixelizeNet

def init_splines(channels: int, 
                 height: int,
                 width: int,
                 hidden_layers: int = 2,
                 hidden_units: int = 8,
                 n_flows: int = 16,
                 ) -> nf.NormalizingFlow:
    # Define flows

    input_shape = (channels, height, width)

    # Set up flows, distributions and merge operations
    q0 = nf.distributions.base.Uniform(input_shape, -1, 1)
    flows = []
    
    for j in range(n_flows):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels=channels, 
                                                                 num_blocks=hidden_layers, 
                                                                 num_hidden_channels=hidden_units)]
        flows += [nf.flows.LULinearPermute(channels)]
    # Construct flow model with the multiscale architecture
    model = nf.NormalizingFlow(q0, 
                               flows, 
                               q0)
    return model

def init_glow(channels: int, 
              hidden_channels: int,
              n_flows: int,
              height: int, 
              width: int,
              scale: bool = True,
              scale_map: Literal["sigmoid", "exp"] = "sigmoid",
              ) -> nf.NormalizingFlow:
    # Define flows

    input_shape = (channels, height, width)

    # Set up flows, distributions and merge operations
    q0 = nf.distributions.base.Uniform(input_shape, 0, 1)
    flows = []
    
    for j in range(n_flows):
        flows += [nf.flows.GlowBlock(channels, hidden_channels,
                                    split_mode='channel', 
                                    scale_map=scale_map, leaky=0.01,
                                    scale=scale, net_actnorm=False)]
    # Construct flow model with the multiscale architecture
    model = nf.NormalizingFlow(q0, 
                               flows, 
                               q0)
    return model

def init_realnvp(channels: int, 
                 height: int,
                 width: int,
                 hidden_units: int = 8,
                 n_flows: int = 6,
                 output_fn: str = None,
                 output_scale: str = None,
                 ) -> nf.NormalizingFlow:
    # Define flows
    input_shape = (channels, height, width)

    # Set up flows, distributions and merge operations
    q0 = nf.distributions.base.Uniform(input_shape, -1, 1)
    flows = []
    

    def binary(x: torch.Tensor, bits: int):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=torch.bool)

    # Coupling masks, count binary for full coverage () w.r.t channels / latent size. Only possibly if channels is small
    _masks = binary(torch.arange(1, 2 ** channels - 1), channels).to(dtype=torch.uint8)
    _expressive_shape = _masks.shape[0]

    rep, crop = divmod(n_flows, _expressive_shape)

    masks = torch.zeros((n_flows, channels), dtype=torch.uint8)
    if rep > 0:
        masks[:rep * _expressive_shape] = _masks.repeat((rep, 1))
    masks[rep * _expressive_shape:] = _masks[:crop]

    flows = []

    for i in range(n_flows):
        s = nf.nets.MLP([channels, hidden_units, channels], init_zeros=True, output_fn = output_fn, output_scale=output_scale)
        t = nf.nets.MLP([channels, hidden_units, channels], init_zeros=True, output_fn = output_fn, output_scale=output_scale)
        flows += [nf.flows.MaskedAffineFlow(masks[i], t, s)]
        flows += [nf.flows.ActNorm(channels)]
    
    
    # Construct flow model with the multiscale architecture
    model = nf.NormalizingFlow(q0, 
                               flows, 
                               q0)
    return model

def get_norm(norm: Literal["minmax", "meanstd"], **kwargs) -> Transform:
    if norm == "minmax":
        return MinMax(**kwargs)
    elif norm == "meanstd":
        return MeanStd(**kwargs)
    else:
        raise ValueError("Invalid norm")

def real_nvp_path_connected_net(
        channels: int = 2,
        hidden_units: int = 130,
        flow_n_flows: int = 6,
        flow_output_fn: Optional[str] = None,
        flow_output_scale: Optional[float] = None,
        norm: Literal["minmax", "meanstd"] = "minmax",
        spatial_shape: tuple = (1000, 1000),
        convex_net_hidden_units: int = 130,
        convex_net_hidden_layers: int = 2,
        dtype: torch.dtype = torch.float32,
        network_type: Optional[Type[PathConnectedNet]] = None,
        network_args: Optional[Dict[str, Any]] = None,
        **kwargs) -> PathConnectedNet:
    
    if network_type is None:
        network_type = PathConnectedNet
    
    if network_args is None:
        network_args = dict()

    if norm == "meanstd"and channels > 2:
        logging.warning("MeanStd normalization is not tested for realnvp for more than 2 channels.")
    

    flow_net = init_realnvp(
                        channels=channels, 
                        hidden_units=hidden_units,
                        output_fn=flow_output_fn,
                        output_scale=flow_output_scale,
                        n_flows=flow_n_flows,
                        height=spatial_shape[0], 
                        width=spatial_shape[1]
                        )

    norm_grid = network_type.create_normalized_grid(grid_shape=spatial_shape if channels == 2 else (100, *spatial_shape))
    
    if len(norm_grid.shape) == 3:
        norm_grid = norm_grid.unsqueeze(0)
    
    norm = get_norm(norm, dim=(0, 2, 3)) # Channel wise normalization
    norm.fit(norm_grid)
    
    norm_flow = NormNet(net=PixelizeNet(flow_net), norm=norm)
    path_net = network_type(convex_net=ConvexNextNet(
                                n_hidden=convex_net_hidden_units, 
                                n_hidden_layers=convex_net_hidden_layers,
                                in_features=channels), 
                                flow_net=norm_flow, 
                                in_channels=channels,
                                **network_args
                                )
    return path_net