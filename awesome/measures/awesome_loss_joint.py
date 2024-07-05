
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Tuple, Union, Dict, Optional, Set
from enum import Enum
import math

class AwesomeLossJoint():

    def __init__(self,
                 criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = 1, 
                 name: str = None,
                 scribble_percentage: float = 1.,
                    **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.BCELoss() if criterion is None else criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.extra_penalty: bool = False
        self.scribble_percentage = scribble_percentage
        self.logger = None
        self.tracker = None

    def __ignore_on_iter__(self) -> Set[str]:
        ret = set()
        ret.add("logger")
        ret.add("tracker")
        return ret

    def random_indices(self, output: torch.Tensor) -> torch.Tensor:
        total = output.shape[-2]
        n_scribbles = int(math.floor(total * self.scribble_percentage))
        random = total - n_scribbles
        return n_scribbles, random

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        n_scribbles, random = self.random_indices(output)

        output_seg = output[..., :n_scribbles, 0][..., None]
        output_convx = output[..., :n_scribbles, 1][..., None]

        seg_loss = self.criterion(output_seg, target)
        self.criterion.apply_gradient_penalty = False
        prior_loss = self.criterion(output_convx, target)
        self.criterion.apply_gradient_penalty = True

        loss = seg_loss + self.alpha * prior_loss
        self.logger.summary_writer.add_scalar(
                self.__class__.__name__+"/seg_loss",
                seg_loss,
                global_step=self.tracker.global_steps)
        
        
        
        self.logger.summary_writer.add_scalar(
                self.__class__.__name__+"/prior_loss",
                prior_loss,
                global_step=self.tracker.global_steps)

        if self.extra_penalty and random > 0:
            output_seg_random = output[..., random:, 0][..., None]
            output_convx_random = output[..., random:, 1][..., None]
            loss = self.gamma * loss
            # Align on random points:
            penalty_loss = torch.mean((output_convx_random - output_seg_random)**2)
            loss += self.beta * penalty_loss
            self.logger.summary_writer.add_scalar(
                self.__class__.__name__+"/penalty_loss",
                penalty_loss,
                global_step=self.tracker.global_steps)
            # Align on scribbles:
            #loss += torch.mean((output_convx - (output_seg).float())**2)

        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
