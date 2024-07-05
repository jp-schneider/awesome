
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Tuple, Union, Dict, Optional, Set
from enum import Enum
import math
import logging 
from awesome.measures.torch_reducable_metric import TorchReducableMetric

class WeightedLoss(TorchReducableMetric):

    def __init__(self,
                 criterion: nn.Module = None,
                 noneclass: Optional[float] = None,
                 name: Optional[str] = None,
                 forward_kwargs_criterion: bool = False,
                 reduction: Literal["sum", "mean", "none"] = "mean",
                 reduction_dim: Optional[Tuple[int, ...]] = None,
                 mode: Literal["none", "sssdms", "equal"] = "none",
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(name=name, reduction=reduction, reduction_dim=reduction_dim)
        if decoding:
            return
        self.name = name
        self.criterion = criterion
        self.forward_kwargs_criterion = forward_kwargs_criterion
        self.noneclass = noneclass
        if criterion is None:
            raise ValueError("criterion must be specified")
        # Set reduction in 
        if hasattr(self.criterion, "reduction"):
            self.criterion.reduction = "none"
        else:
            logging.warning("Criterion has no reduction attribute")
        self.mode = mode

    def _compute_weight(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.mode == "sssdms":
            # Implementation of the 
            # Count the number of samples per class
            vals, counts = torch.unique(target, return_counts=True)
            fg_count, bg_count = counts[vals == 0], counts[vals == 1]
            ratio = bg_count / fg_count
            ratio /= 10
            ratio = torch.round(ratio)
            ratio += 1
            # Compute the weight
            weight = torch.ones_like(target)
            weight[target == 0] = ratio
            # Apply the weight
            return weight
        elif self.mode == "equal":
            vals, counts = torch.unique(target, return_counts=True)
            fg_count, bg_count = counts[vals == 0], counts[vals == 1]
            ratio = bg_count / fg_count
            # Compute the weight
            weight = torch.ones_like(target)
            weight[target == 0] = ratio
            return weight
        else:
            raise ValueError(f"Mode {self.mode} is not supported")

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # 
        t = target
        o = output
        # Remove noneclass
        if self.noneclass is not None:
            o = output[target != self.noneclass]
            t = target[target != self.noneclass]
        if len(t.shape) >= 2:
            # Reshape the output to B x C for the criterion
            if len(t.shape) == 3:
                raise ValueError("Expected 4D target, got 3D target")
            # Permute channel to last dim and reshape B, C, H, W to B, C
            t_shape = (t.shape[0] * t.shape[2] * t.shape[3], t.shape[1])
            o = o.permute(0, 2, 3, 1).reshape(t_shape)
            t = t.permute(0, 2, 3, 1).reshape(t_shape)
        # Apply criterion
        args = dict()
        if self.forward_kwargs_criterion:
            args = kwargs
        loss = self.criterion(o, t, **args)
        # Apply weights
        if self.mode != "none":
            weight = self._compute_weight(o, t, **kwargs)
            loss = loss * weight
        return self.compute_return_value(loss)