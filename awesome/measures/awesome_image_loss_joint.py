
import torch.nn as nn
import torch
from awesome.measures.tracker_loss import TrackerLoss

from awesome.util.torch import tensorify
from typing import List, Literal, Set, Tuple, Union, Dict, Optional
from enum import Enum
import math

class AwesomeImageLossJoint(TrackerLoss):

    def __init__(self,
                 criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = 1,
                 name: str = None,
                 map_initially_on_segmentation: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = nn.BCELoss() if criterion is None else criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.extra_penalty: bool = False
        self.map_initially_on_segmentation = map_initially_on_segmentation
        self.logger = None
        self.tracker = None


    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        output_seg = output[:, :output.shape[1] // 2, ...]
        output_convx = output[:, output.shape[1] // 2:, ...]

        seg_loss = self.criterion(output_seg, target, **kwargs)

        self.criterion.apply_gradient_penalty = False
        prior_loss = self.criterion(output_convx, target, **kwargs)
        self.criterion.apply_gradient_penalty = True

        loss = seg_loss + self.alpha * prior_loss
        self.log("segmentation_loss", seg_loss)
        
    
        if self.extra_penalty:
            loss = self.gamma * loss
            # Align on all points
            penalty_loss = torch.mean((output_convx - output_seg)**2)
            loss += self.beta * penalty_loss
            self.log("penalty_loss", penalty_loss)

        else:
            if self.map_initially_on_segmentation:
                loss = self.gamma * loss
                # Align on all points
                penalty_loss = torch.mean((output_convx - (output_seg > 0.5).float())**2)
                loss += self.beta * penalty_loss
                self.log("penalty_loss", penalty_loss)
            
        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
