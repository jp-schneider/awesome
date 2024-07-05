
from typing import Any, Callable
import torch 
from awesome.util.torch import TensorUtil
import math

def pixelize(keep: bool = False):
    """Acts as a decorator for a model forward function where the input tensors are expected to be of shape (B, C, H, W).
    The decorator will reshape the input to (B * H * W, C) and reshape the output back to (B, C, H, W).

    The decorator will not alter anything with the shape is unequal to (B, C, H, W).
    
    Parameters
    ----------
    keep : bool, optional
        If set to true, it will not reshape the output, by default False
    """
    def _outer_wrapper(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapper(*args, **kwargs):
            nonlocal keep
            # Inner wrapper which will be called when original function is invoked.
            # Assuming input is of shape (B, C, H, W)
            # if not just do nothing
            apply = False
            b, h, w = None, None, None
            ref_input = args[1]
            if len(ref_input.shape) == 4:
                apply = True
                b, c, h, w = ref_input.shape

            def _reshape(x: torch.Tensor):
                v = x.permute(0, 2, 3, 1)
                return v.reshape((v[..., -1].numel(), v.shape[-1]))
            
            def _reshape_back(x: torch.Tensor):
                v = x.reshape((b, h, w, -1))
                return v.permute(0, 3, 1, 2)
            
            new_args = args
            new_kwargs = kwargs

            if apply:
                # Skip first argument (self)
                new_args = TensorUtil.apply_deep(args[1:], fnc=_reshape)
                new_args = (args[0],) + new_args
                new_kwargs = TensorUtil.apply_deep(kwargs, fnc=_reshape)
            
            out = function(*new_args, **new_kwargs)

            if apply and (not keep):
                out = TensorUtil.apply_deep(out, fnc=_reshape_back)
            return out
        return wrapper
    return _outer_wrapper