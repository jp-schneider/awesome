
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset as TorchDataset
from awesome.dataset.base_dataset import BaseDataset
from awesome.dataset.batched_dataset import BatchedDataset
from awesome.dataset.separable_dataset import SeparableDataset


class TorchDataSource(BaseDataset, TorchDataset, SeparableDataset, BatchedDataset):
    """Base class for datasets implemented for the torch agent.Adds functionality of retriving index."""

    def __init__(self,
                 returns_index: bool = True,
                 decoding: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.returns_index = returns_index

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns the item at the given index.

        Parameters
        ----------
        index : int
            Index for the requested data tuple.

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]
            Returns a tuple containing: (Input-Tensor, Label-Tensor) if not `self.returns_index` 
            else:
                (Input-Tensor, Label-Tensor, Index-Tensor)
        """
        raise NotImplementedError()
