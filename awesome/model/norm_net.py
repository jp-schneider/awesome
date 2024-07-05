import torch
from awesome.transforms.fittable_transform import FittableTransform
from awesome.transforms.invertable_transform import InvertableTransform
from awesome.util.torch import TensorUtil

class NormNet(torch.nn.Module):
    """Takes a network and normalizes the input and output with a given normalization."""

    def __init__(self, 
                 net: torch.nn.Module, 
                 norm: torch.nn.Module
                 ):
        super().__init__()
        self.net = net
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fit if necessary
        if isinstance(self.norm, FittableTransform) and not self.norm.fitted:
            self.norm.fit(x)
        # Normalize input
        x = self.norm.transform(x)
        # Forward pass
        x = self.net(x)
        # Normalize output
        x = self.norm.inverse_transform(x)
        return x

    def reset_parameters(self) -> None:
        return TensorUtil.reset_parameters(self.net)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Fit if necessary
        if isinstance(self.norm, FittableTransform) and not self.norm.fitted:
            raise ValueError("Norm must be fitted before inverting.")
        # Normalize input
        x = self.norm.transform(x)
        # Forward pass
        x = self.net.inverse(x)
        # Normalize output
        x = self.norm.inverse_transform(x)
        return x