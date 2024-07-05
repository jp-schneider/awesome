from typing import Any
import torch
from abc import ABC, abstractmethod

from awesome.transforms.transform import Transform


class FittableTransform(Transform):
    """Abstract class for fittable transform. 
    This transform can be fitted to the data. Can only be executed after fitting."""

    def __init__(self) -> None:
        super().__init__()
        self.fitted = False

    def fit(self, *args, **kwargs) -> None:
        """Abstract method for fitting the transform to the data.

        Parameters
        ----------
        *args : Any
            Any positional arguments.
        **kwargs : Any
            Any keyword arguments.
        """
        self.fitted = True

    def transform(self, *args, **kwargs) -> Any:
        """Transforms the data.

        Parameters
        ----------
        *args : Any
            Any positional arguments.
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        Any
            The transformed data.
        """
        if not self.fitted:
            raise RuntimeError("Transform must be fitted before it can be executed.")
        pass
    
    def fit_transform(self, *args, **kwargs) -> Any:
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    
