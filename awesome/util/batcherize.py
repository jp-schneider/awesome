
from typing import Any, Callable
import torch 

def batcherize(keep: bool = False, expected_dim: int = 4):
    """Adds batch dimension to a input of a model when used as decorator.
    if input is of shape smaller than expected dim, it append dimensions to the biginnen until reaching expected_dim.

    Functionality can be disabled, if the original function is called with a 'no_extend = True' key-word argument. 

    Parameters
    ----------
    keep : bool, optional
        If the batch information / dimension should be keeped on return if added, by default False

    expected_dim : int optional
        The number of expected dimension of the input
    """
    def _outer_wrapper(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapper(*args, **kwargs):
            nonlocal keep
            # Inner wrapper which will be called when original function is invoked.
            batch_added = False
            added_dims = tuple()

            new_args = []
            
            # Check if should extend
            no_extend = kwargs.pop("no_extend", False)

            if not no_extend:
                for i, v in enumerate(args):
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) < expected_dim:
                            to_add = tuple([None for _ in range(expected_dim - len(v.shape))]) + (Ellipsis, )
                            added_dims = tuple([0 for _ in to_add if _ is None]) + (Ellipsis, )
                            v = v[to_add]
                            batch_added = True
                    new_args.append(v)
            else:
                new_args = args

            out = function(*new_args, **kwargs)

            if batch_added and (not keep) and len(out.shape) >= len(added_dims):
                out = out[added_dims]
            return out
        return wrapper
    return _outer_wrapper