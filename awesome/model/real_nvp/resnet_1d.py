import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Model altered from: https://github.com/chrischute/real-nvp/

def weights_init_normal(activation: str = 'relu'):
    if not isinstance(activation, str):
        raise ValueError('activation must be a string')
    def _weights_init_normal(m: torch.nn.Module):
        nonlocal activation
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)
                gain = nn.init.calculate_gain(activation, 0)
                fan = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                std = gain / math.sqrt(fan)
                if m.bias is not None:
                    m.bias.data.uniform_(-std, std)
    return _weights_init_normal

def weights_init_uniform(activation: str = 'relu'):
    if not isinstance(activation, str):
        raise ValueError('activation must be a string')
    def _weights_init_uniform(m: torch.nn.Module):
        nonlocal activation
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=activation)
                gain = nn.init.calculate_gain(activation, 0)
                fan = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                std = gain / math.sqrt(fan)
                if m.bias is not None:
                    m.bias.data.uniform_(-std, std)
    return _weights_init_uniform

class WNLinear(nn.Module):
    """Weight-normalized linear layer."""

    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 bias=True,
                 **kwargs):
        super(WNLinear, self).__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_channels, out_channels, bias=bias), dim=None)

    def reset_parameters(self, activation: str = 'relu'):
        with torch.no_grad():
            self.linear.weight_g.data.fill_(1)
            gain = nn.init.calculate_gain(activation, 0)
            fan = nn.init._calculate_correct_fan(self.linear.weight_v, 'fan_in')
            std = gain / math.sqrt(fan)
            nn.init.kaiming_uniform_(self.linear.weight_v, mode='fan_in', nonlinearity=activation)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-std, std)


    def forward(self, x):
        x = self.linear(x)
        return x


class ResidualBlock1D(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels:int = 1,
                 **kwargs
                   ):
        super(ResidualBlock1D, self).__init__()

        self.in_norm = nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.in_linear = WNLinear(in_channels, out_channels, bias=False)

        self.out_norm = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.out_linear = WNLinear(out_channels, out_channels,  bias=True)
        self.in_linear.apply(weights_init_normal('relu'))
        self.out_linear.apply(weights_init_normal('relu'))


    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_linear(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_linear(x)

        x = x + skip
        return x


class ResNet1D(nn.Module):
    """1D ResNet for scale and translate factors in 1D Real NVP."""
    
    def __init__(self, 
                 in_channels: int = 2,
                 mid_channels: int = 128, 
                 out_channels: int = 2,
                 num_blocks: int = 2, 
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
        super(ResNet1D, self).__init__()
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
        self.out_linear.apply(weights_init_normal('linear'))


    def forward(self, x):
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
        return x
