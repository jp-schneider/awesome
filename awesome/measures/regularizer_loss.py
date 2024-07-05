
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import Any, Callable, List, Literal, Tuple, Union, Dict, Optional
from enum import Enum
import math

class RegularizerLoss():

    def __init__(self,
                 criterion: nn.Module = None,
                 tau: float = 0.,
                 regularizer: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
                 name: str = None,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        if decoding:
            return
        self.name = name
        self.criterion = criterion
        if criterion is None:
            raise ValueError("criterion must not be None")
        self.tau = tau
        self.regularizer = regularizer
        if regularizer is None and self.tau > 0.:
            raise ValueError("regularizer must not be None if tau is larger zero!")
        
    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = self.criterion(output, target)
        if self.tau > 0.:
            regularizer = self.regularizer(output, **kwargs)
            loss += self.tau * regularizer
        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
