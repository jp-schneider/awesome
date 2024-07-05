from typing import Any, Optional, Tuple, Union
import torch

from awesome.transforms.fittable_transform import FittableTransform
from awesome.transforms.invertable_transform import InvertableTransform



def minmax(v: torch.Tensor,
           v_min: Optional[torch.Tensor] = None,
           v_max: Optional[torch.Tensor] = None,
           new_min: torch.Tensor = 0.,
           new_max: torch.Tensor = 1.,
           ) -> torch.Tensor:
    if v_min is None:
        v_min = torch.min(v)
    if v_max is None:
        v_max = torch.max(v)
    return (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min

class MinMax(torch.nn.Module, InvertableTransform, FittableTransform):
    """MinMax normalization."""
    def __init__(self, 
                 new_min: torch.Tensor = -1, 
                 new_max: torch.Tensor = 1,
                 dim: Optional[Union[int, Tuple[int]]] = None
                 ):
        super().__init__()
        self.register_buffer("min", torch.zeros(1))
        self.register_buffer("max", torch.ones(1))
        self.register_buffer("new_min", torch.tensor(new_min))
        self.register_buffer("new_max", torch.tensor(new_max))
        self.dim = dim

    def _reduce(self, x: torch.Tensor, op: Any, dim: Any) -> torch.Tensor:
        if dim is None or isinstance(dim, int):
            x = op(x, dim=dim)
            if not isinstance(x, torch.Tensor):
                x = x.values
            return x
        else:
            for d in dim:
                x = op(x, dim=d, keepdim=True)
                if not isinstance(x, torch.Tensor):
                    x = x.values
            return x

    def fit(self, x: torch.Tensor):
        super().fit(x)
        self.min = self._reduce(x, torch.min, self.dim)
        self.max = self._reduce(x, torch.max, self.dim)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        super().transform(x)
        return minmax(x, self.min, self.max, self.new_min, self.new_max)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return minmax(x, self.new_min, self.new_max, self.min, self.max)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"