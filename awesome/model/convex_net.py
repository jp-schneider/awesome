import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awesome.model.real_nvp.resnet_1d import WNLinear, weights_init_uniform, weights_init_normal

from awesome.util.pixelize import pixelize


class ConvexNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_channels: int = 2,
                 **kwargs):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.W0y = nn.Linear(in_channels, n_hidden)
        self.W1z = nn.Linear(n_hidden, n_hidden)
        self.W2z = nn.Linear(n_hidden, 1)

        # something skippy
        self.W1y = nn.Linear(in_channels, n_hidden, bias=False)
        self.W2y = nn.Linear(in_channels, 1, bias=False)

    @pixelize()
    def forward(self, x):
        # define forward pass
        # Input of shape (batch_size, 2)
        x_input = x
        x = F.relu(self.W0y(x))
        x = F.relu(self.W1z(x) + self.W1y(x_input))
        x = self.W2z(x) + self.W2y(x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            self.W1z.weight.data = F.relu(self.W1z.weight.data)
            self.W2z.weight.data = F.relu(self.W2z.weight.data)


class WNSkipBlock(nn.Module):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 130,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__()
        self.ln = WNLinear(in_features, out_features)
        self.skp = WNLinear(in_skip_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return F.relu(self.ln(x) + self.skp(x_input))

    def reset_parameters(self) -> None:
        self.ln.reset_parameters('relu')
        self.skp.reset_parameters('relu')

    def enforce_convexity(self) -> None:
        self.ln.linear.weight_v.data = F.relu(self.ln.linear.weight_v.data)
        self.ln.linear.weight_g.data = F.relu(self.ln.linear.weight_g.data)

        self.skp.linear.weight_v.data = F.relu(self.skp.linear.weight_v.data)
        self.skp.linear.weight_g.data = F.relu(self.skp.linear.weight_g.data)


class WNOutBlock(WNSkipBlock):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 1,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            in_skip_features=in_skip_features
        )

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return self.ln(x) + self.skp(x_input)

    def reset_parameters(self) -> None:
        self.ln.reset_parameters('linear')
        self.skp.reset_parameters('linear')

# Does not work .... 
class WNConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 ** kwargs):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.input = WNLinear(in_features, n_hidden)
        self.skip = nn.ModuleList([
                        WNSkipBlock(in_features=n_hidden, 
                                  out_features=n_hidden, 
                                  in_skip_features=in_features) for _ in range(n_hidden_layers)])
        self.out = WNOutBlock(
            in_features=n_hidden, 
            out_features=out_features, 
            in_skip_features=in_features)

    def reset_parameters(self) -> None:
        self.input.reset_parameters("linear")
        for i in range(len(self.skip)):
            self.skip[i].reset_parameters()
        self.out.reset_parameters()

    @pixelize()
    def forward(self, x):
        # define forward pass
        # Input of shape (batch_size, 2)
        x_input = x
        x = F.relu(self.input(x))
        for i in range(len(self.skip)):
            x = self.skip[i](x, x_input=x_input)
        x = self.out(x, x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            for i in range(len(self.skip)):
                self.skip[i].enforce_convexity()
            self.out.enforce_convexity()



class SkipBlock(nn.Module):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 130,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)
        self.skp = nn.Linear(in_skip_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return F.relu(self.ln(x) + self.skp(x_input))

    def reset_parameters(self) -> None:
        self.ln.apply(weights_init_uniform('relu'))
        self.skp.apply(weights_init_uniform('relu'))

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            self.ln.weight.data = F.relu(self.ln.weight.data)
            #self.skp.weight.data = F.relu(self.skp.weight.data)



class OutBlock(SkipBlock):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 1,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            in_skip_features=in_skip_features
        )

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return self.ln(x) + self.skp(x_input)

    def reset_parameters(self) -> None:
        self.ln.apply(weights_init_uniform('linear'))
        self.skp.apply(weights_init_uniform('linear'))

class ConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 ** kwargs):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.input = nn.Linear(in_features, n_hidden)
        self.skip = nn.ModuleList([
                        SkipBlock(in_features=n_hidden, 
                                  out_features=n_hidden, 
                                  in_skip_features=in_features) for _ in range(n_hidden_layers)])
        self.out = OutBlock(
            in_features=n_hidden, 
            out_features=out_features, 
            in_skip_features=in_features)

    def reset_parameters(self) -> None:
        self.input.apply(weights_init_uniform('linear'))
        for i in range(len(self.skip)):
            self.skip[i].reset_parameters()
        self.out.reset_parameters()
        return True

    @pixelize()
    def forward(self, x):
        # define forward pass
        # Input of shape (batch_size, 2)
        x_input = x
        x = F.relu(self.input(x))
        for i in range(len(self.skip)):
            x = self.skip[i](x, x_input=x_input)
        x = self.out(x, x_input=x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            for i in range(len(self.skip)):
                self.skip[i].enforce_convexity()
            self.out.enforce_convexity()