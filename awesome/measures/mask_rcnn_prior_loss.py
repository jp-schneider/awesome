
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Set, Tuple, Union, Dict, Optional, Any
from enum import Enum
import math

class MaskRcnnPriorLoss():

    def __init__(self,
                 prior_criterion: nn.Module = None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = 1,
                 name: str = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
        self.prior_criterion = nn.BCELoss() if prior_criterion is None else prior_criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.extra_penalty: bool = False
        # self.logger = None
        # self.tracker = None

    def __ignore_on_iter__(self) -> Set[str]:
        ret = set()
        ret.add("logger")
        ret.add("tracker")
        return ret

    def __call__(self, output: List[Dict[str, Any]], target: torch.Tensor, **kwargs) -> torch.Tensor:
        # Enumerate over batch images, evaluate loss for each image
        img_losses = torch.zeros(len(output), device=output[0]['detections'][0]['masks'].device)

        for i, item in enumerate(output):
            # Losses of mask rcnn per image
            seg_loss = sum((v for k, v in item.items() if 'loss' in k))
            
            if len(item['detections']) == 0 or item['detections'][0]['masks'].numel() == 0:
                # No detections, no loss for the prior
                img_losses[i] = seg_loss
                continue
            
            # Should be a tensor containing a mask for each instance B x C x H x W, where B is the object
            seg_masks = item['detections'][0]['masks']
            prior_masks = item['prior']
            seg_masks_sig = torch.sigmoid(seg_masks)

            # Fit prior around prediction 
            if not self.extra_penalty:
                seg_masks_sig = (seg_masks_sig >= 0.5).float() # Maskrcnn returns mask as 1 for true, 0 for false invert these
            else:
                # Maskrcnn returns mask as 1 for true, 0 for false invert these
                seg_masks_sig = 1 - seg_masks_sig

            penalty_loss = torch.mean((prior_masks - seg_masks_sig)**2)
            seg_loss += self.alpha * penalty_loss

            img_losses[i] = seg_loss

        return torch.mean(img_losses)





    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
