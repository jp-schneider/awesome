import torch
import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Union
from awesome.measures.torch_metric import TorchMetric, VEC_TYPE
from abc import ABC, abstractmethod
import copy


class TorchReducableMetric(TorchMetric):
    """A reducable metric where different reductions can be defined."""

    REDUCTIONS = {
        "sum": torch.sum,
        "mean": torch.mean,
        "none": lambda x: x,
        "max": torch.max,
        "min": torch.min,
    }

    def __init__(self,
                 reduction: Literal["sum", "mean", "none"] = "mean",
                 reduction_dim: Optional[Tuple[int, ...]] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(name=name)
        if reduction not in TorchReducableMetric.REDUCTIONS:
            raise ValueError(
                f"Value {reduction} for reduction is invalid. Supported values are: {','.join(TorchReducableMetric.REDUCTIONS.keys())}")
        self.reduction = reduction
        self.reduction_dim = reduction_dim

    @classmethod
    def reduce_value(cls, _input: torch.Tensor, reduction: Literal["sum", "mean", "none", "max", "min"], **kwargs) -> torch.Tensor:
        if reduction == "none":
            return _input
        return TorchReducableMetric.REDUCTIONS[reduction](_input, **kwargs)

    def compute_return_value(self, _loss: torch.Tensor) -> torch.Tensor:
        """Computes the return value based on a per-element / sample loss function.
        So it is responsible for calculating the per-batch loss and metrics.

        Parameters
        ----------
        _loss : torch.Tensor
            The loss per sample / element

        Returns
        -------
        torch.Tensor
            A tensor dependent on reduction and dimm.
        """
        args = dict()
        if self.reduction_dim is not None:
            args["dim"] = self.reduction_dim
        return TorchReducableMetric.reduce_value(_loss, self.reduction, **args)

    @abstractmethod
    def __call__(self, output: VEC_TYPE,
                 target: VEC_TYPE, **kwargs) -> torch.Tensor:
        """Computes the current metric based on the output and target data.
        Return value depends on `reduce` and the dimension.

        Parameters
        ----------
        output : Union[torch.Tensor, np.ndarray]
            The output of a model.
        target : Union[torch.Tensor, np.ndarray]
            The target or the ground truth.

        Returns
        -------
        Union[torch.Tensor, StatisticTensor]
            Resulting loss.
        """
        raise NotImplementedError()
