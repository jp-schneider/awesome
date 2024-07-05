from typing import Literal, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awesome.model.real_nvp.resnet_1d import ResNet1D, ResidualBlock1D, WNLinear, weights_init_normal, weights_init_uniform


def capped_exp(x: torch.Tensor, thresh: torch.Tensor = torch.tensor(5.), slope: float = 1e-2) -> torch.Tensor:
    """Capped exponential function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    thresh : torch.Tensor, optional
        Threshold at which to cap the exponential function, by default 5.
    slope : float, optional
        Slope of the linear part of the capped exponential function, by default 1e-2.

    Returns
    -------
    torch.Tensor
        Capped exponential function applied to the input tensor.
    """
    return torch.where(x < thresh, torch.exp(x), torch.exp(thresh) + slope * x)


class DiffeomorphismNet(nn.Module):
    def __init__(self, **kwargs):
        # call constructor from superclass
        super().__init__()
        self.l1a_scale = nn.Linear(1, 50)
        self.l1b_scale = nn.Linear(50, 1)

        self.l1a_bias = nn.Linear(1, 50)
        self.l1b_bias = nn.Linear(50, 1)

        self.l2a_scale = nn.Linear(1, 50)
        self.l2b_scale = nn.Linear(50, 1)

        self.l2a_bias = nn.Linear(1, 50)
        self.l2b_bias = nn.Linear(50, 1)

        # self.final = nn.Linear(2,2)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1b_scale.weight.data.fill_(0.0)
        self.l1b_bias.weight.data.fill_(0.0)
        self.l2b_scale.weight.data.fill_(0.0)
        self.l2b_bias.weight.data.fill_(0.0)

    def forward(self, x):
        # define forward pass

        _1st_lin = self.l1a_scale(x[:, 0].view(-1, 1))
        _1st_bias = self.l1a_bias(x[:, 0].view(-1, 1))

        s = self.l1b_scale(F.relu(_1st_lin))
        t = self.l1b_bias(F.relu(_1st_bias))

        cap_s = capped_exp(s)

        xx = x[:, 1] * cap_s.view(-1) + t.view(-1)

        _2nd_lin = self.l2a_scale(xx.view(-1, 1))
        _2nd_bias = self.l2a_bias(xx.view(-1, 1))

        ss = self.l2b_scale(F.relu(_2nd_lin))
        tt = self.l2b_bias(F.relu(_2nd_bias))

        cap_ss = capped_exp(ss)

        yy = x[:, 0] * cap_ss.view(-1) + tt.view(-1)

        # return self.final(torch.concat((xx.view(-1,1),yy.view(-1,1)), axis=1))
        ret = torch.concat((xx.view(-1, 1), yy.view(-1, 1)), axis=1)
        return ret


class SimpleBackbone(nn.Module):

    def __init__(self,
                 in_channels: int = 2,
                 network_width: int = 10,
                 **kwargs) -> None:
        super().__init__()
        self.linear1 = WNLinear(in_channels, network_width)
        self.linear2 = WNLinear(network_width, in_channels)
        #self.linear1.apply(weights_init_normal('relu'))
        #self.linear2.apply(weights_init_normal('linear'))

    def reset_parameters(self) -> None:
        self.linear1.apply(weights_init_uniform('relu'))
        self.linear2.apply(weights_init_uniform('tanh'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x


class SimpleResnet(nn.Module):

    def __init__(self, 
                 in_channels: int = 2,
                 mid_channels: int = 128, 
                 out_channels: int = 2,
                 num_blocks: int = 1, 
                 double_after_norm: bool = False
                 ):
        """1D ResNet for scale and translate factors in 1D Real NVP.


        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 2
        mid_channels : int, optional
            Number if intermediate channels, by default 128
        out_channels : int, optional
            Number of output channels, by default 2
        num_blocks : int, optional
            Number of residual blocks, by default 2
        double_after_norm : bool, optional
            If the channel values should be doubled after norming the input, by default False
        """
        super(SimpleResnet, self).__init__()
        self.in_norm = nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.double_after_norm = double_after_norm
        self.in_linear = WNLinear(2 * in_channels, mid_channels, bias=True)
        self.in_skip = WNLinear(mid_channels, mid_channels, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock1D(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNLinear(mid_channels, mid_channels, bias=True)
                                    for _ in range(num_blocks)])
        self.out_norm = nn.BatchNorm1d(mid_channels, track_running_stats=False)
        self.out_linear = WNLinear(mid_channels, out_channels, bias=True)
        
        self.in_linear.apply(weights_init_normal('relu'))
        self.in_skip.apply(weights_init_normal('relu'))
        self.skips.apply(weights_init_normal('relu'))
        self.out_linear.apply(weights_init_normal('tanh'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        if self.double_after_norm:
            x *= 2.
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        x = self.in_linear(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_linear(x)
        x = torch.tanh(x)
        return x


class NormalBlock(nn.Module):
    """Basic block with weight norm."""

    def __init__(self, 
                 in_channels: int = 1,
                 mid_channels: int = 128,
                 out_channels:int = 1,
                 **kwargs
                   ):
        super(NormalBlock, self).__init__()
        self.in_linear = WNLinear(in_channels, mid_channels, bias=True)
        self.out_linear = WNLinear(mid_channels, out_channels,  bias=True)


    def reset_parameters(self) -> None:
        self.in_linear.apply(weights_init_uniform('leaky_relu'))
        self.out_linear.apply(weights_init_uniform('tanh'))

    def forward(self, x):
        x = self.in_linear(x)
        x = F.leaky_relu(x)
        x = self.out_linear(x)
        x = torch.tanh(x)
        return x

def simple_backbone(input_width, network_width=10):

    return nn.Sequential(
            nn.Linear(input_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, input_width),
            # nn.ReLU(),
            # nn.Linear(network_width, input_width),
            # nn.ReLU(),
            # nn.Linear(network_width, input_width),
            nn.Tanh(),
    )


class WNScale(nn.Module):

    weight: torch.Tensor

    def __init__(self, dim: int = 1, **kwargs) -> None:
        super().__init__()
        self.scale = nn.utils.weight_norm(nn.Linear(dim, dim))
        self.weights_init_normal(self.scale)
        self.weight = nn.Parameter(torch.tensor(
            [1.0 + 0.01 * torch.randn((1, ))]))

    def reset_parameters(self) -> None:
        self.weights_init_normal(self.scale)
        with torch.no_grad():
            self.weight.data = torch.tensor([1.0 + 0.01 * torch.randn((1, ))], 
                                                dtype=self.weight.dtype, 
                                                device=self.weight.device)
            
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.scale(self.weight)

    def weights_init_normal(self, m):
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)


class NormalizingFlow1D(nn.Module):

    def __init__(self,
                 num_coupling: int = 4,
                 width: int = 130,
                 num_blocks: int = 1,
                 in_features: int = 2,
                 backbone: Literal['default', 'residual_block', 'resnet'] = 'default',
                 **kwargs
                 ):
        super(NormalizingFlow1D, self).__init__()
        self.num_coupling = num_coupling
        if self.num_coupling % in_features != 0:
            raise ValueError(
                f'Number of coupling layers should be divisible by in_features ({in_features})')
        
        _backbone: Union[SimpleBackbone, ResNet1D] = None
        args = dict(in_channels=1)
        if backbone == 'default':
            _backbone = SimpleBackbone
            args['network_width'] = width
        elif backbone == 'resnet':
            _backbone = SimpleResnet
            args['mid_channels'] = width
            args['out_channels'] = args['in_channels']
            args['num_blocks'] = num_blocks
        elif backbone == 'residual_block' or backbone == 'normal_block':
            _backbone = NormalBlock
            args['mid_channels'] = width
            args['out_channels'] = args['in_channels']
            #args['num_blocks'] = num_blocks

        else:
            raise ValueError(f'Unknown backbone: {backbone}')
        
        self.in_features = in_features
        self.s = nn.ModuleList([_backbone(**args)
                               for x in range(num_coupling)])
        self.t = nn.ModuleList([_backbone(**args)
                               for x in range(num_coupling)])
        # Learnable scaling parameters for outputs of S
        self.scale = nn.ModuleList([WNScale(dim=1)
                                   for x in range(num_coupling)])

    def reset_parameters(self) -> None:
        for s, t, scale in zip(self.s, self.t, self.scale):
            s.reset_parameters()
            t.reset_parameters()
            scale.reset_parameters()
        return True

    def forward(self, x):
        # s_vals = []
        x1, x2 = x[:, :1], x[:, 1:]
        for i in range(self.num_coupling):
            # Alternating which var gets transformed
            if i % 2 == 0:
                s = self.scale[i]() * self.s[i](x1)
                x2 = torch.exp(s) * x2 + self.t[i](x1)
            else:
                s = self.scale[i]() * self.s[i](x2)
                x1 = torch.exp(s) * x1 + self.t[i](x2)
            # s_vals.append(s)

        # Return outputs and vars needed for determinant
        return torch.cat([x1, x2], 1)  # , torch.cat(s_vals)

NormalizingFlow2D = NormalizingFlow1D
