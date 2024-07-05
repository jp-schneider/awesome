
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
from typing import Any, Callable, Generator, List, Mapping, Optional, Tuple, Type

import torch

from awesome.model.number_based_multi_prior_module import NumberBasedMultiPriorModule
from awesome.model.abstract_multi_prior_module import AbstractMultiPriorModule


class BatchSizeMultiPriorModule(NumberBasedMultiPriorModule):
    """Multi prior module which contains a prior for each batch item. The number of priors is determined by the batch size."""

    def forward(self, *args, batch_size: int, **kwargs) -> torch.Tensor:
        fwd = self.prior_type.forward
        return self._wrapped(fwd, num_priors=batch_size, *args, **kwargs)

            
    