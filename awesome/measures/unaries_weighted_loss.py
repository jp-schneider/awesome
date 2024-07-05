
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
from awesome.measures.weighted_loss import WeightedLoss
from torch import Tensor


class UnariesWeightedLoss(WeightedLoss):
    """Same as weighted loss, but treats targets as unaries, eg not as classes and weights"""
    
    def __init__(self, 
                criterion: nn.Module = None, 
                noneclass: Optional[str] = None, 
                name: Optional[str] = None, 
                forward_kwargs_criterion: bool = False, 
                reduction: Literal['sum', 'mean', 'none'] = "mean", 
                reduction_dim: Optional[Tuple[int, ...]] = None, 
                mode: Literal['none', 'equal', 'ratio', 'sssdms'] = "none", 
                ratio: float = 1.,
                decoding: bool = False, **kwargs):
        super().__init__(
            criterion=criterion,
            noneclass=noneclass,
            name=name,
            forward_kwargs_criterion=forward_kwargs_criterion,
            reduction=reduction,
            reduction_dim=reduction_dim,
            mode=mode,
            decoding=decoding,
            **kwargs
        )
        self.ratio = ratio

    def _compute_weight(self, output: Tensor, target: Tensor, **kwargs) -> Tensor:
        if self.mode == "ratio":
            # Assuming B x C x H x W where c = 1 
            vals, counts = torch.unique(target >= 0.5, return_counts=True)
            fg_count, bg_count = counts[vals == 0], counts[vals == 1]
            class_correction = bg_count / fg_count
            # Compute the weight
            weight = torch.ones_like(target)
            weight[target < 0.5] = ((class_correction - 1) * self.ratio) + 1
            return weight
        elif self.mode == "sssdms":
            # Implementation of the Loss in 
            # Count the number of samples per class
            vals, counts = torch.unique(target >= 0.5, return_counts=True)
            fg_count, bg_count = counts[vals == 0], counts[vals == 1]
            ratio = bg_count / fg_count
            ratio /= 10
            ratio = torch.round(ratio)
            ratio += 1
            # Compute the weight
            weight = torch.ones_like(target)
            weight[target < 0.5] = ratio
            # Apply the weight
            return weight
        elif self.mode == "equal":
            # Assuming B x C x H x W where c = 1 
            vals, counts = torch.unique(target >= 0.5, return_counts=True)
            fg_count, bg_count = counts[vals == 0], counts[vals == 1]
            class_correction = bg_count / fg_count
            # Compute the weight
            weight = torch.ones_like(target)
            weight[target < 0.5] = class_correction
            return weight
        else:
            raise ValueError(f"Mode {self.mode} is not supported")