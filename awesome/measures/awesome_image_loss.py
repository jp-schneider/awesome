
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Tuple, Union, Dict, Optional
from enum import Enum
import math

class AwesomeImageLoss():

    def __init__(self,
                 criterion: nn.Module = None,
                 prior_criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 100.,
                 gamma: float = 0.1,
                 name: str = None,
                 forward_kwargs_criterion: bool = True,
                 forward_kwargs_prior_criterion: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.BCELoss() if criterion is None else criterion
        self.prior_criterion = nn.BCELoss() if prior_criterion is None else prior_criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.extra_penalty: bool = False
        self.forward_kwargs_criterion = forward_kwargs_criterion
        self.forward_kwargs_prior_criterion = forward_kwargs_prior_criterion

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        output_seg = output[:, :output.shape[1] // 2, ...]
        output_convx = output[:, output.shape[1] // 2:, ...]

        args = dict()
        if self.forward_kwargs_criterion:
            args = kwargs
        loss = self.criterion(output_seg, target, **args)

        args = dict()
        if self.forward_kwargs_prior_criterion:
            args = kwargs
        loss += self.alpha * self.prior_criterion(output_convx, target, **args)

        if self.extra_penalty:
            loss = self.gamma * loss
            # Align on all points
            loss += self.beta * torch.mean((output_convx - (output_seg > 0.5).float())**2)
        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
