import torch
import numpy as np
from typing import Any, Optional, TypeVar, Union
from abc import abstractmethod
from enum import Enum

from awesome.util.torch import VEC_TYPE

class TorchMetric():
    """Metric which can be used to evaluate the performance of a model."""

    def __init__(self,
                 name: Optional[str] = None
                 ) -> None:
        super().__init__()
        self.name = name
        """Some alternative name for the loss. By default it will be the class name."""

    @abstractmethod
    def __call__(self, output: VEC_TYPE, target: VEC_TYPE) -> torch.Tensor:
        """Computes the current metric based on the output and target data.

        Parameters
        ----------
        output : Union[torch.Tensor, np.ndarray]
            The output of a model.
        target : Union[torch.Tensor, np.ndarray]
            The target or the ground truth.

        Returns
        -------
        torch.Tensor
            The calculated metric.
        """
        raise NotImplementedError()

    def get_name(self) -> str:
        """Returns the name of the metric.

        Returns
        -------
        str
            The name, typically the class.
        """
        if self.name is not None:
            return self.name
        return type(self).__name__
