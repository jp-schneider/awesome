from typing import Any
import torch
from abc import ABC, abstractmethod

from awesome.transforms.transform import Transform


class InvertableTransform(Transform):
    """Abstract class for invertable transform."""
    
    @abstractmethod
    def inverse_transform(self, *args, **kwargs) -> Any:
        """Abstract method for implementing the inverse transform.

        Parameters
        ----------
        *args : Any
            Any positional arguments.
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        Any
            The inverse transformed data.
        """
        pass
    
    def __call__(self, *args, inverse: bool = False, **kwargs) -> Any:
        """Calls the transform.

        Parameters
        ----------
        *args : Any
            Any positional arguments.

        inverse : bool, optional
            If the inverse transform should be used, by default False
        
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        Any
            The transformed data.
        """
        if inverse:
            return self.inverse_transform(*args, **kwargs)
        else:
            return self.transform(*args, **kwargs)
