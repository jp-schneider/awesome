
import torch
from awesome.measures.torch_metric import TorchMetric
from awesome.measures.torch_reducable_metric import TorchReducableMetric

from typing import Literal, Tuple, Optional

class UnariesConversionLoss(TorchMetric):

    def __init__(self,
                 criterion: torch.nn.Module = None,
                 name: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(
            name=name)
        self.criterion = criterion


    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        target = (target >= 0.5).float()
        return self.criterion(output, target, **kwargs)
    
    def get_name(self) -> str:
        if self.name is None:
            return "UC" + self.criterion.get_name()
        return self.name