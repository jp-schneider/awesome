
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Tuple, Union, Dict, Optional
from enum import Enum
import math

class PriorImageLoss():

    def __init__(self,
                 criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 100.,
                 delta: float = 1,
                 noneclass: int = 2,
                 name: str = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.BCELoss() if criterion is None else criterion
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.noneclass = noneclass 
        self.extra_penalty: bool = False

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        output_seg = output[:, :output.shape[1] // 2, ...]
        output_convx = output[:, output.shape[1] // 2:, ...]

        if self.noneclass is not None:
            output_seg_no_noneclass = output_seg[target != self.noneclass]
            output_convx_no_noneclass = output_convx[target != self.noneclass]
            target = target[target != self.noneclass]

        loss = (self.delta * self.criterion(output_seg_no_noneclass, target)) + (self.alpha * self.criterion(output_convx_no_noneclass, target))

        # Align on all points (Also those which are noneclass)
        loss += self.beta * torch.mean((output_convx - (output_seg > 0.5).float())**2)

        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
