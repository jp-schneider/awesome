
import torch.nn as nn
import torch

from typing import Literal, Tuple, Union, Dict, Optional
from .torch_metric import TorchMetric, VEC_TYPE
from awesome.util.torch import tensorify
import logging

class PixelAccuracy(TorchMetric):
    """Calculates the pixel accuracy, e.g. percentage of correctly predicted pixels."""

    def __init__(
            self,
            noneclass: Optional[int] = None,
            name: str = None, **kwargs) -> None:
        """Creates a new instance of the pixel accuracy metric.

        Parameters
        ----------
        noneclass : Optional[int], optional
            Noneclass which should be ignored, by default None
        name : str, optional
            Name of the loss by default PixelAccuracy, by default None
        """
        super().__init__(name=name)
        self.noneclass = noneclass


    def __call__(self, output: VEC_TYPE, target: VEC_TYPE, **kwargs) -> torch.Tensor:
        target = tensorify(target)
        output = tensorify(output)
        _len = 1.
        if self.noneclass is not None:
            mask = (target != self.noneclass)
            target = target[mask]
            output = output[mask]
        _len = torch.numel(target)
        bin = (output == target).float()
        if _len == 0.:
            logging.warning("No valid pixels found. Avoiding division by zero. Accuracy will be inaccurate.")
            _len = 1.
        return torch.sum(bin) / _len
