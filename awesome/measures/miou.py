
import torch.nn as nn
import torch

from typing import Literal, Tuple, Union, Dict, Optional
from awesome.measures.torch_metric import TorchMetric, VEC_TYPE
from sklearn.metrics import jaccard_score
from awesome.util.torch import tensorify

class MIOU(TorchMetric):
    """Calculates the intersection over union (IOU) for a given output and target, and mean over all images. Calculates it only w.r.t one class."""

    def __init__(
            self,
            noneclass: Optional[int] = None,
            noneclass_replacement: Optional[int] = None,
            average: Optional[Literal["binary", "macro", "micro", "weighted", "samples"]] = "binary",
            invert: bool = False,
            name: str = None, **kwargs) -> None:
        super().__init__(name=name)
        self.noneclass = noneclass
        self.noneclass_replacement = noneclass_replacement
        self.average = average
        self.invert = invert
        if self.noneclass is not None and self.noneclass_replacement is None:
            self.noneclass_replacement = 0


    def __call__(self, output: VEC_TYPE, target: VEC_TYPE, **kwargs) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu()
        t = target
        o = output
        device = o.device if isinstance(o, torch.Tensor) else torch.device("cpu")
        if self.noneclass is not None:
            t = torch.where(t == self.noneclass, self.noneclass_replacement, t)
            # Changed to ignore noneclass in output also by target
            o = torch.where(t == self.noneclass, self.noneclass_replacement, o)
        if self.invert:
            t = 1. - t
            o = 1. - o
        t = t.squeeze().reshape(t.numel())
        o = o.squeeze().reshape(t.numel())
        if torch.all(t == 0.):
            return torch.tensor(0., dtype=torch.float32, device=device)
        iou = tensorify(jaccard_score(t, o, average=self.average), dtype=torch.float32, device=device)
        return iou
