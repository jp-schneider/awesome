from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseDataset(ABC):
    """Abstract dataset definition for all datasets."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Returns an item at the given index.

        Parameters
        ----------
        index : int
            The index of an item to retrieve.

        Returns
        -------
        Any
            The value at the index.
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the current dataset.

        Returns
        -------
        int
            The number of total items. Items are zero based (0, len(self) - 1)
        """
        ...

    def set_config(self, **kwargs) -> None:
        """
        Writes any arbitrary parameters to the current object.

        Parameters
        ----------
        **kwargs
            Any key-value pair will be written to attribtes named accordingly.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the config parameters for the current dataset / source.

        Returns
        -------
        Dict[str, Any]
            Returns all values captured in self.vars().
        """
        return self.vars()

    def vars(self) -> Dict[str, Any]:
        """
        Returning all non private properties with their values, which are not private (dunder) fields.

        Returns
        -------
        Dict[str, Any]
            Dictionary of property name and its value.
        """
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}
