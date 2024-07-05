from typing import Dict, List, Optional, Tuple, Union
from awesome.dataset.torch_datasource import TorchDataSource
from abc import abstractmethod
import numpy as np


class _NOSUBSET():
    pass

NOSUBSET = _NOSUBSET()

class SubdivisibleDataset(TorchDataSource):
    """Base class for datasets which can be subdivided into a subset based on indices, selections etc."""

    __subset_indices__: Optional[Dict[int, int]]
    """Maps the 0 Based subset index to the index of the original dataset."""

    __subset_indices_reversed__: Optional[Dict[int, int]]
    """Maps the index of the original dataset to the 0 Based subset index. Will contain only the indices which are part of the subset."""

    __subset_specifier__: Optional[Union[int, List[int], slice]]
    """The subset specifier which was used to create the subset indices. 
    Can be a list of strings, a slice or a anything which can be used to slice a list. Contains the value of the subset init parameter."""

    @property
    def subset_indices(self) -> Union[Dict[int, int], _NOSUBSET]:
        """Returns the subset indices if they are available, otherwise NOSUBSET.

        Returns
        -------
        Union[Dict[int, int], _NOSUBSET]
            s. above
        """
        if self.__subset_indices__ is None:
            if self.__subset_specifier__ is None:
                self.__subset_indices__ = NOSUBSET
            else:
                self.__subset_indices__ = self.create_subset_mapping()
                self.__subset_indices_reversed__ = None
        return self.__subset_indices__

    @property
    def subset_indices_reversed(self) -> Union[Dict[int, int], _NOSUBSET]:
        if self.__subset_indices_reversed__ is None:
            mapping = self.subset_indices
            if mapping == NOSUBSET:
                self.__subset_indices_reversed__ = mapping
            else:
                self.__subset_indices_reversed__ = {v: k for k, v in mapping.items()}
        return self.__subset_indices_reversed__

    def __init__(self, 
                 subset: Optional[Union[int, List[int], slice]] = None,
                 returns_index: bool = True, 
                 **kwargs) -> None:
        super().__init__(returns_index, **kwargs)
        self.__subset_specifier__ = subset
        self.__subset_indices__ = None
        self.__subset_indices_reversed__ = None

    @abstractmethod
    def create_subset_mapping(self) -> Dict[int, int]:
        """Creates a mapping from the 0 Based subset index to the index of the original dataset.
        Must be overridden by subclasses.

        Returns
        -------
        Dict[int, int]
            A mapping from the subset index to the original dataset index.
        """        
        raise NotImplementedError()


    def _subset_split_indices(self, train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        subset = self.subset_indices
        if subset == NOSUBSET:
            return train, val
        in_train = list(map(lambda x: (x[0], x[1] in train), subset.items()))
        trains = [x[0] for x in in_train if x[1]]
        vals = [x[0] for x in in_train if not x[1]]
        return np.array(trains), np.array(vals)
    
    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        train, val = super().split_indices()
        return self._subset_split_indices(train, val)

    def get_data_index(self, index: int) -> int:
        """Gets the index of the original dataset from the subset index.

        Parameters
        ----------
        index : int
            The subset index.

        Returns
        -------
        int
            The index of the original dataset.
        """
        if self.subset_indices == NOSUBSET:
            return index
        return self.subset_indices[index]

    def subset_len(self) -> int:
        if self.subset_indices == NOSUBSET:
            return NOSUBSET
        return len(self.subset_indices)

    def get_subset_index(self, data_index: int) -> Optional[int]:
        """Gets the subset index from the original dataset index.
        Entries which are not part of the subset will return None.

        Parameters
        ----------
        data_index : int
            The index of the original dataset.

        Returns
        -------
        Optional[int]
            The subset index if the entry is part of the subset, otherwise None.
        """        
        if self.subset_indices_reversed == NOSUBSET:
            return data_index
        return self.subset_indices_reversed.get(data_index, None)