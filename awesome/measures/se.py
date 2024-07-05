
import torch
from awesome.measures.torch_reducable_metric import TorchReducableMetric

from typing import Literal, Tuple, Optional

class SE(TorchReducableMetric):
    """Squared Error loss class. Computes the squared error between the output and the target."""
    def __init__(self,
                 reduction: Literal["sum", "mean", "none"] = "mean",
                 reduction_dim: Optional[Tuple[int, ...]] = None,
                 name: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(
            reduction=reduction, 
            reduction_dim=reduction_dim, 
            name=name)


    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        se = (target - output)**2
        return self.compute_return_value(se)
    
    def get_name(self) -> str:
        if self.name is None:
            reduction = (self.reduction[0] if self.reduction is not None else "").upper()
            return reduction + "SE"
        return self.name