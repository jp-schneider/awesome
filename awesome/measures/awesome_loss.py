
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Tuple, Union, Dict, Optional
from enum import Enum
import math
import inspect

class AwesomeLoss():

    def __init__(self,
                 criterion: nn.Module = None,
                 alpha: float = 1.,
                 name: str = None,
                 scribble_percentage: float = 1.,
                    **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.BCELoss() if criterion is None else criterion
        self.alpha = alpha
        self.extra_penalty: bool = False
        self.scribble_percentage = scribble_percentage
        self._forward_kwargs = None

    @property
    def forward_kwargs(self) -> bool:
        if self._forward_kwargs is None:
            if isinstance(self.criterion, nn.Module):
                self._forward_kwargs = ('kwargs' in inspect.signature(self.criterion.forward).parameters)
            else:
                self._forward_kwargs = ('kwargs' in inspect.signature(self.criterion.__call__).parameters)
        return self._forward_kwargs
    
    def random_indices(self, output: torch.Tensor) -> torch.Tensor:
        total = output.shape[-2]
        n_scribbles = int(math.floor(total * self.scribble_percentage))
        random = total - n_scribbles
        return n_scribbles, random



    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        n_scribbles, random = self.random_indices(output)

        output_seg = output[..., :n_scribbles, 0][..., None]
        output_convx = output[..., :n_scribbles, 1][..., None]

        child_loss_args = dict()
        if self.forward_kwargs:
            child_loss_args.update(kwargs)

        loss = self.criterion(output_seg, target, **child_loss_args) + self.alpha * self.criterion(output_convx, target, **child_loss_args)

        if self.extra_penalty and random > 0:
            output_seg_random = output[..., random:, 0][..., None]
            output_convx_random = output[..., random:, 1][..., None]
            loss = 0.1 * loss
            # Align on random points:
            loss += 100 * torch.mean((output_convx_random - (output_seg_random > 0.5).float())**2)
            
        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
