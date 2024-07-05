

from functools import wraps
from os import PathLike
from typing import BinaryIO, Callable, Dict, List, Optional, Tuple, Type, Any, Union
import torch

from awesome.util.prior_cache import PriorCache
from awesome.util.reflection import class_name, dynamic_import
from awesome.serialization import JsonConvertible
from awesome.util.torch import TensorUtil
from torch.utils.data import default_collate
def prior():
    """This is a decorator for the __getitem__ of a dataset.
    it can be used to return the prior of an item if it is available.
    """
    def decorator(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(function)
        def wrapper(*args, **kwargs):
            self = args[0]
            item = args[1]
            return_prior = self.return_prior
            out = function(*args, **kwargs)
            if self.has_prior and return_prior:
                _state = self.__prior_cache__[item]
                _prior = (item, _state)
                return _prior, out
            else:
                return out
        return wrapper
    return decorator

def create_prior_collate_fn(
        has_prior,
        inner_collate_fn: Optional[Callable[[Any], Any]] = None,
        ) -> Callable[[Any], Any]:
    """Creates a collate function which is used to collate the data of a dataset.
    If the dataset has a prior cache, the state is extracted and returned as a list, while the rest is handled by the inner collate function.

    Parameters
    ----------
    has_prior : bool
        Wether the dataset has a prior cache.
        And the prior is prepended to each actual item within a batch.
    inner_collate_fn : Optional[Callable[[Any], Any]], optional
        Inner collate function, by default the default_collate function is used.
    
    Returns
    -------
    Callable[[Any], Any]
        Collate function.
    """
    if inner_collate_fn is None:
        inner_collate_fn = default_collate
    def collate_fn(batch: List[Tuple[Any, Tuple[Any, ...]]])-> Union[Tuple[List[Any], Any], Any]:
        nonlocal has_prior
        nonlocal inner_collate_fn
        if not has_prior:
            return inner_collate_fn(batch)
        else:
            # Extract the state and return it as list, while the rest is handled by the inner collate function.
            states = []
            items = []
            for state, item in batch:
                items.append(item)
                states.append(state)
            return states, inner_collate_fn(items)
    return collate_fn
        
class PriorManager():
    """Context manager which is initialized by the model and applies """


    def __init__(self, 
                 model: torch.nn.Module, 
                 prior_state: Tuple[int, Any] = None,
                 prior_cache: Union[PriorCache, 'PriorDataset'] = None,
                 model_device: Optional[torch.device] = None,
                 store_device: Optional[torch.device] = None,
                 training: bool = False,
                 ) -> None:
        self.model = model
        self.state = prior_state
        if isinstance(prior_cache, PriorDataset):
            prior_cache = prior_cache.__prior_cache__
        if not isinstance(prior_cache, PriorCache):
            prior_cache = None
        self.prior_cache = prior_cache
        if store_device is None and prior_cache is not None:
            self.prior_cache.store_device = store_device
        if model_device is None:
            model_device = next(self.model.parameters()).device
        self.model_device = model_device
        self.training = training

    def __enter__(self) -> None:
        if self.state is None or self.prior_cache is None:
            return
        key, state = self.state
        if self.prior_cache is not None and self.model_device != self.prior_cache.store_device:
            state = TensorUtil.apply_deep(state, fnc=lambda x: x.to(device=self.model_device))
        PriorCache.apply_prior(self.model, state)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.state is None or self.prior_cache is None:
            return False
        key, state = self.state
        ext_prior = PriorCache.extract_prior(self.model)
        self.prior_cache[key] = ext_prior
        return False

class PriorDataset():
    """If the dataset needs a prior cache, this mixin allows for the management of the prior cache."""

    def __init__(self, 
                 prior_model_type: Type[torch.nn.Module] = None,
                 prior_model_args: Dict[str, Any] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.__has_prior__ = False
        self.return_prior = True
        """Controls wether the decorator on getitem returns the prior or not."""
        if prior_model_args is None:
            prior_model_args = dict()
        if prior_model_type is not None:
            self.__has_prior__ = True
            self.__prior_cache__ = PriorCache(prior_model_type, prior_model_args)
        else:
            self.__prior_cache__ = None

    @property
    def has_prior(self) -> bool:
        return self.__has_prior__


    def prior_save(self, f: Union[PathLike[str], BinaryIO]) -> None:
        """Saves the prior cache to the given file / buffer.

        Parameters
        ----------
        f : Union[PathLike[str], BinaryIO]
            File / Buffer
        """
        if self.has_prior:
            return self.__prior_cache__.save(f)


    def prior_load(self, f: Union[PathLike[str], BinaryIO]):
        """Loads the prior cache from the given file / buffer.

        Parameters
        ----------
        f : Union[PathLike[str], BinaryIO]
            File / Buffer
        """
        cache = PriorCache.load(f)
        self.__prior_cache__ = cache

    