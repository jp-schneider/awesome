
import torch
from awesome.measures.torch_reducable_metric import TorchReducableMetric

from typing import Literal, Tuple, Optional

class AE(TorchReducableMetric):

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
        ae = torch.abs(target - output)
        return self.compute_return_value(ae)
    
    def get_name(self) -> str:
        if self.name is None:
            reduction = (self.reduction[0] if self.reduction is not None else "").upper()
            return reduction + "AE"
        return self.name