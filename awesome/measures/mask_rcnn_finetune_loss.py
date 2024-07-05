
import torch.nn as nn
import torch

from awesome.util.torch import tensorify
from typing import List, Literal, Set, Tuple, Union, Dict, Optional, Any
from enum import Enum
import math

class MaskRcnnFinetuneLoss():

    def __init__(self,
                 name: str = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name
    

    def __call__(self, output: List[Dict[str, Any]], target: torch.Tensor, **kwargs) -> torch.Tensor:
        # Enumerate over batch images, evaluate loss for each image
        img_losses = torch.zeros(len(output))

        for i, item in enumerate(output):
            # Losses of mask rcnn per image
            seg_loss = torch.sum(torch.stack([v for k, v in item.items() if 'loss' in k]))
            if img_losses.device != seg_loss.device:
                img_losses = img_losses.to(device=seg_loss.device)
            img_losses[i] = seg_loss

        return torch.mean(img_losses)

    def __ignore_on_iter__(self) -> Set[str]:
        ret = set()
        ret.add("logger")
        ret.add("tracker")
        return ret


    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
