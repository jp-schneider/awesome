
import torch.nn as nn
import torch
from awesome.measures.tracker_loss import TrackerLoss
from awesome.measures.unaries_weighted_loss import UnariesWeightedLoss

from awesome.util.torch import tensorify
from typing import List, Literal, Set, Tuple, Union, Dict, Optional
from enum import Enum
import math
from awesome.measures.se import SE

class FBMSJointLoss(TrackerLoss):

    def __init__(self,
                 criterion: nn.Module = None,
                 penalty_criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 clip_penalty: bool = True,
                 name: str = None,
                 **kwargs
                 ) -> None:
        super().__init__(name=name)
        self.name = name
        self.criterion = UnariesWeightedLoss(nn.BCELoss(), mode="sssdms") if criterion is None else criterion
        self.penalty_criterion = SE(reduction="mean") if penalty_criterion is None else penalty_criterion
        self.alpha = alpha
        self.beta = beta
        self.clip_penalty = clip_penalty
        self.logger = None
        self.tracker = None


    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        
        output_seg = output[:, :output.shape[1] // 2, ...]
        output_convx = output[:, output.shape[1] // 2:, ...]

        seg_loss_raw = self.criterion(output_seg, target, **kwargs)

        seg_loss = self.alpha * seg_loss_raw
        self.log("segmentation_loss", seg_loss_raw)
        
        # Align on all points
        penalty_loss_raw = self.penalty_criterion(output_convx, output_seg)
        penalty_loss = self.beta * penalty_loss_raw
        
        if self.clip_penalty:
            # Performing soft clip the penalty loss so it can not be too large => Normal segmentation should be dominant
            if penalty_loss > seg_loss:
                penalty_loss = penalty_loss * (seg_loss / penalty_loss).detach()
        
        self.log("penalty_loss", penalty_loss_raw)
        loss = seg_loss + penalty_loss

        self.log("penalty_loss_frac", penalty_loss / loss)
        self.log("segmentation_loss_frac", seg_loss / loss)
        return loss
