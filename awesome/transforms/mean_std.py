from typing import Any
import torch

from awesome.transforms.fittable_transform import FittableTransform
from awesome.transforms.invertable_transform import InvertableTransform

class MeanStd(InvertableTransform, FittableTransform, torch.nn.Module):
    """Mean standard deviation normalization."""

    def __init__(self, dim: Any = None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))
        self.dim=dim

    def fit(self, x: torch.Tensor):
        super().fit(x)
        self.mean = x.mean(dim=self.dim, keepdim=True)
        self.std = x.std(dim=self.dim, keepdim=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        super().transform(x)
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x : torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"
