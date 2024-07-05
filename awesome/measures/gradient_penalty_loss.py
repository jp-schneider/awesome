
import torch.nn as nn
import torch
from awesome.measures.tracker_loss import TrackerLoss

from awesome.util.torch import tensorify
from typing import Any, Callable, List, Literal, Tuple, Union, Dict, Optional
from enum import Enum
import math

class GradientPenaltyLoss(TrackerLoss):

    def __init__(self,
                 criterion: nn.Module = None,
                 apply_gradient_penalty: bool = False,
                 xygrad: float = 0.,
                 rgbgrad: float = 0.,
                 featgrad: float = 0.,
                 xytype: Literal["xy", "feat", "featxy", "edge", "edgexy"] = "xy",
                 noneclass: Optional[float] = None,
                 name: str = None,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.criterion = criterion
        self.xygrad = xygrad
        self.rgbgrad = rgbgrad
        self.featgrad = featgrad
        self.xytype = xytype
        if self.xytype not in ["xy", "feat", "featxy", "edge"]:
            raise ValueError(f"xytype must be one of [xy, feat, featxy, edge] but is {self.xytype}")
        self.apply_gradient_penalty = apply_gradient_penalty
        self.noneclass = noneclass
        if decoding:
            return
        if criterion is None:
            raise ValueError("criterion must not be None")
        
        
        
    def __call__(self, 
                 output: torch.Tensor, 
                 target: torch.Tensor, 
                 _input: Tuple[Any] = None,
                 **kwargs) -> torch.Tensor:
        # Ignore the output if target contains the noneclass
        original_output = output
        if self.noneclass is not None:
            output = output[target != self.noneclass]
            target = target[target != self.noneclass]
        loss = self.criterion(output, target, **kwargs)

        if self.apply_gradient_penalty:
            if _input is None:
                raise ValueError("GradientPenaltyLoss needs _input to apply gradient penalty")
            img = _input[0]
            raw_xy = _input[1] # xy is the coordinates they maybe combined with semantic features, so there this must be considered

            # Summing up the output to a scalar value, as the gradient would be the pixelwise gradient then
            output_sum = torch.sum(original_output)

            if self.xygrad > 0. or self.featgrad > 0.:            
                grad_raw_xy= torch.autograd.grad(output_sum, raw_xy, retain_graph=True, create_graph=True)[0] # Index 0 as result is a tuple of size 1
                
                grad_feat = None
                grad_xy = None
                
                mean_feat = None
                mean_xy = None
                # Splitting the gradient with respect to the xy coordinates / semantic features / edges based on the used type and regularize them
                if self.xytype == "feat":
                    grad_feat = grad_raw_xy
                    mean_feat = torch.mean(torch.abs(grad_feat))
                elif self.xytype == "xy":
                    grad_xy = grad_raw_xy
                    mean_xy = torch.mean(torch.abs(grad_xy))
                elif self.xytype == "featxy":
                    grad_xy = grad_raw_xy[:, :2, ...]
                    mean_xy = torch.mean(torch.abs(grad_xy))

                    grad_feat = grad_raw_xy[:, 2:, ...] # Semantic features are the last channels
                    mean_feat = torch.mean(torch.abs(grad_feat))
                elif self.xytype == "edge":
                    # Doesnt supported yet.
                    pass
                elif self.xytype == "edgexy":
                    grad_xy = grad_raw_xy[:, :2, ...]
                    mean_xy = torch.mean(torch.abs(grad_xy))
                else:
                    raise ValueError(f"xytype must be one of [xy, feat, featxy] but is {self.xytype}")

                if self.xygrad > 0. and mean_xy is not None:
                    xy_grad_loss = self.xygrad * mean_xy
                    
                    loss += xy_grad_loss
                    self.log("xy_grad_loss", xy_grad_loss)
                    
                if self.featgrad > 0. and mean_feat is not None:
                    feat_grad_loss = self.featgrad * mean_feat
                    loss += feat_grad_loss
                    self.log("feat_grad_loss", feat_grad_loss)
            
            if self.rgbgrad > 0.:
                grad_rgb = torch.autograd.grad(output_sum, img, retain_graph=True, create_graph=True)[0]
                mean_rgb = torch.mean(torch.abs(grad_rgb))
                rgb_grad_loss = self.rgbgrad * mean_rgb
                loss += rgb_grad_loss
                self.log("rgb_grad_loss", rgb_grad_loss)

        return loss

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
