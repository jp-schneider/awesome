
from typing import Any, Callable
import torch 
import numpy as np

def process_value(value: Any, expected_dim: int = 3) -> Any:
    added_dims = tuple()
    channel_added = False
    if isinstance(value, torch.Tensor):
        if len(value.shape) < expected_dim:
            to_add = tuple([None for _ in range(expected_dim - len(value.shape))]) + (Ellipsis, )
            added_dims = tuple([0 for _ in to_add]) + (Ellipsis, )
            value = value[to_add]
            channel_added = True
    if isinstance(value, np.ndarray):
        if len(value.shape) < expected_dim:
            to_add = tuple([None for _ in range(expected_dim - len(value.shape))]) + (Ellipsis, )
            added_dims =  (Ellipsis, ) + tuple([0 for _ in to_add])
            value = value[to_add]
            channel_added = True
    return value, added_dims, channel_added

def channelize(keep: bool = False):
    """Adds channel dimension to a input of a model when used as decorator.
    Assumes that the input is of shape ([batch x channels x ] x H x W), or ([batch x] x H x W [x channels]) (resp. torch.Tensor or np.ndarray) 
    and adds a channel dimension if not present. Pytorch gets channel dimension added at the beginning, numpy at the end.

    if input is of shape smaller than expected dim, it append dimensions to the beginning / end until reaching expected_dim.

    Functionality can be disabled, if the original function is called with a 'no_extend = True' key-word argument. 

    Parameters
    ----------
    keep : bool, optional
        If the batch information / dimension should be keeped on return if added, by default False
    """
    def _outer_wrapper(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapper(*args, **kwargs):
            nonlocal keep
            # Inner wrapper which will be called when original function is invoked.
            channel_added = False
            added_dims = tuple()

            new_args = []
            
            # Check if should extend
            no_extend = kwargs.pop("no_extend", False)
            new_kwargs = dict(kwargs)

            if not no_extend:
                for i, v in enumerate(args):
                    v, _added_dims, _channel_added = process_value(v, expected_dim=3, keep=keep)
                    if _channel_added:
                        channel_added = True
                        added_dims = _added_dims
                    new_args.append(v)
                for k, v in kwargs.items():
                    v, _added_dims, _channel_added = process_value(v, expected_dim=3, keep=keep)
                    if _channel_added:
                        channel_added = True
                        added_dims = _added_dims
                    new_kwargs[k] = v
            else:
                new_args = args

            out = function(*new_args, **new_kwargs)

            if channel_added and (not keep) and len(out.shape) >= len(added_dims):
                out = out[added_dims]
            return out
        return wrapper
    return _outer_wrapper