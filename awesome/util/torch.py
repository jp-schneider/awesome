import decimal
import logging
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union
from collections import OrderedDict
import torch
from decimal import Decimal
import numpy as np
from awesome.error import NoIterationTypeError, NoSimpleTypeError, ArgumentNoneError
import hashlib

VEC_TYPE = TypeVar("VEC_TYPE", bound=Union[torch.Tensor, np.ndarray])
"""Vector type, like torch.Tensor or numpy.ndarray."""

NUMERICAL_TYPE = TypeVar(
    "NUMERICAL_TYPE", bound=Union[torch.Tensor, np.generic, int, float, complex, Decimal])
"""Numerical type which can be converted to a tensor."""


def get_weight_normalized_param_groups(network: torch.nn.Module, 
                                       weight_decay: float, 
                                       norm_suffix: str= 'weight_g', 
                                       name_prefix: str=''):
    
    norm_params = []
    unnorm_params = []
    for n, p in network.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    prefix = (name_prefix.strip() + (" " if len(name_prefix.strip()) > 0 else ""))
    param_groups = [{'name': f'{prefix}normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': f'{prefix}unnormalized', 'params': unnorm_params}]
    return param_groups


def tensorify(input: NUMERICAL_TYPE,
              dtype: Optional[torch.dtype] = None,
              device: Optional[torch.device] = None,
              requires_grad: bool = False) -> torch.Tensor:
    """
    Assuring that input is a tensor by converting it to one.
    Accepts tensors or ndarrays.

    Parameters
    ----------
    input : Union[torch.Tensor, np.generic, int, float, complex, Decimal]
        The input

    dtype : Optional[torch.dtype]
        Dtype where input should be belong to. If it differs it will cast the type.
        By default its None and the dtype wont be changed.

    device : Optional[torch.device]
        Device where input should be on / send to. If it differs it will move.
        By default its None and the device wont be changed.

    requires_grad : bool
        If the created tensor requires gradient, Will be only considered if input is not already a tensor!. Defaults to false.

    Returns
    -------
    torch.Tensor
        The created tensor.
    """
    if isinstance(input, torch.Tensor):
        if (dtype and input.dtype != dtype) or (device and input.device != device):
            input = input.to(dtype=dtype, device=device)
        return input
    return torch.tensor(input, dtype=dtype, device=device, requires_grad=requires_grad)


def fourier(x: torch.Tensor) -> torch.Tensor:
    """2D fourier transform with normalization and shift.

    Parameters
    ----------
    x : torch.Tensor
        Spatial data to transform.

    Returns
    -------
    torch.Tensor
        Complex fourier spectrum.
    """
    return torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))


def inverse_fourier(x: torch.Tensor) -> torch.Tensor:
    """2D inverse fourier transform with normalization and shift.

    Parameters
    ----------
    x : torch.Tensor
        Fourier data to transform.

    Returns
    -------
    torch.Tensor
        Spatial output.
    """
    return torch.fft.ifft2(torch.fft.ifftshift(x), norm='forward')


class TensorUtil():
    """Static class for using complex tensor calculations and tensor related utilities"""

    @staticmethod
    def to(object: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Any:
        """Takes an arbitrary object an tries to move all internal tensors to a device and specified dtype.

        Parameters
        ----------
        object : Any
            The object to query.
        dtype : Optional[torch.dtype], optional
            The dtype where tensors should be converted into, None makes no conversion., by default None
        device : Optional[torch.device], optional
            The device where tensors should be moved to, by default None

        Returns
        -------
        Any
            The object with changed properties.

        Raises
        ------
        ArgumentNoneError
            If object is None
        """
        if object is None:
            raise ArgumentNoneError("object")
        if dtype is None and device is None:
            # Do nothing because no operation is specified.
            return object
        return TensorUtil._process_value(object, "", object, dtype=dtype, device=device)


    @staticmethod
    def to_hash(object: Any) -> Any:
        """Takes an object graph and hashes all tensors with sha256.
        Keeps the object graph structure, but replaces all tensors with their sha256 hash.

        Parameters
        ----------
        object : Any
            An arbitrary object graph containing tensors.

        Returns
        -------
        Any
            Object graph with tensors replaced by their sha256 hash.
        """
        def _tensor_hash(x: torch.Tensor) -> str:
            return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        return TensorUtil.apply_deep(object, fnc=_tensor_hash)

    @staticmethod
    def reset_parameters(object: Any, memo: Set[Any] = None) -> None:
        """Resets all parameters of the given object and its children,
        by calling reset_parameters() on supported objects.

        Parameters
        ----------
        object : Any
            The object / module to reset.
        memo : Set[Any], optional
            Memo for already visited objects, by default None
        """        
        if memo is None:
            memo = set()
        try:
            if hasattr(object, "__hash__") and object.__hash__ is not None:
                if object in memo:
                    return
                else:
                    memo.add(object)
        except TypeError as err:
            # Unhashable type
            pass

        if hasattr(object, "reset_parameters"):
            ret = object.reset_parameters()
            if ret is not None and isinstance(ret, bool) and ret:
                # If reset_parameters returns True, it means that it has reset its child parameters
                return ret

        # Recursively reset all parameters, if possible
        if isinstance(object, torch.nn.Module):
            # Proceed with child modules
            for m in object.modules():
                TensorUtil.reset_parameters(m, memo=memo)
        return True

    @staticmethod
    def _process_value(value: Any,
                       name: str,
                       context: Dict[str, Any],
                       dtype: Optional[torch.dtype] = None,
                       device: Optional[torch.device] = None) -> Any:
        try:
            return TensorUtil._process_simple_type(value, name, context, dtype=dtype, device=device)
        except NoSimpleTypeError:
            try:
                return TensorUtil._process_iterable(value, name, context, dtype=dtype, device=device)
            except NoIterationTypeError:
                return value

    @staticmethod
    def _process_simple_type(value, name: str,
                             context: Dict[str, Any],
                             dtype: Optional[torch.dtype] = None,
                             device: Optional[torch.device] = None) -> Any:
        if value is None:
            return value
        elif isinstance(value, torch.Tensor):
            return TensorUtil._process_tensor(value, name, context, dtype=dtype, device=device)
        elif hasattr(value, "to"):
            try:
                return TensorUtil._process_tensor(value, name, context, dtype=dtype, device=device)
            except (TypeError) as err:
                # Propably wrong argument combination
                logging.warning(
                    f"Type Error on invoking 'to' of value: {value}. \n {err}")
                raise NoSimpleTypeError()
        elif isinstance(value, (int, float, str, decimal.Decimal)):
            return value
        else:
            raise NoSimpleTypeError()

    @staticmethod
    def _process_tensor(value: torch.Tensor, name: str,
                        context: Dict[str, Any],
                        dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None
                        ) -> Any:
        return value.to(dtype=dtype, device=device)

    @staticmethod
    def _process_dict(value: Dict[str, Any],
                      name: str,
                      context: Dict[str, Any],
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None) -> Dict[str, Any]:
        # Works same as json hook, creates objects from inside to outside
        ret = {}
        # Handling internals
        for k, v in value.items():
            ret[k] = TensorUtil._process_value(
                v, name, context, dtype=dtype, device=device)
        # Converting ret with hook if childrens are processed
        return ret

    @staticmethod
    def _process_iterable(value, name: str,
                          context: Dict[str, Any],
                          dtype: Optional[torch.dtype] = None,
                          device: Optional[torch.device] = None) -> Any:
        if isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            a = []
            for subval in value:
                a.append(TensorUtil._process_value(
                    subval, name, value, dtype=dtype, device=device))
            if isinstance(value, list):
                return a
            else:
                return tuple(a)
        elif isinstance(value, (dict)):
            return TensorUtil._process_dict(value, name, context, dtype=dtype, device=device)
        elif hasattr(value, '__iter__'):
            # Handling iterables which are not lists or tuples => handle them as dict.
            return TensorUtil._process_dict(dict(value), name, context, dtype=dtype, device=device)
        elif hasattr(value, '__dict__'):
            new_val = TensorUtil._process_dict(
                dict(value.__dict__), name, context, dtype=dtype, device=device)
            # Setting all properties manually
            for k, v in new_val.items():
                setattr(value, k, v)
        else:
            raise NoIterationTypeError()

    @staticmethod
    def apply_deep(obj: Any, fnc: Callable[[torch.Tensor], torch.Tensor], memo: Set[Any] = None) -> Any:
        """Applies the given function on each tensor found in a object graph.

        Creates a deep copy of each object in the graph while querying it.

        Parameters
        ----------
        obj : Any
            A object containing tensors.
        fnc : Callable[[torch.Tensor], torch.Tensor]
            A function to apply for.
        memo : Set[Any], optional
            Memo of already visitend objects., by default None

        Returns
        -------
        Any
            The altered object.
        """
        if memo is None:
            memo = set()
        try:
            if hasattr(obj, "__hash__") and obj.__hash__ is not None:
                if obj in memo:
                    return obj
                else:
                    memo.add(obj)
        except TypeError as err:
            # Unhashable type
            pass
        if isinstance(obj, (str, int, float, complex)):
            return obj
        if isinstance(obj, torch.Tensor):
            ret = fnc(obj)
            return ret
        elif isinstance(obj, list):
            return [TensorUtil.apply_deep(x, fnc=fnc, memo=memo) for x in obj]
        elif isinstance(obj, tuple):
            vals = [TensorUtil.apply_deep(x, fnc=fnc, memo=memo) for x in obj]
            return tuple(vals)
        elif isinstance(obj, set):
            return set([TensorUtil.apply_deep(x, fnc=fnc, memo=memo) for x in obj])
        elif isinstance(obj, OrderedDict):
            return OrderedDict({k: TensorUtil.apply_deep(v, fnc=fnc, memo=memo) for k, v in obj.items()})
        elif isinstance(obj, dict):
            return {
                k: TensorUtil.apply_deep(v, fnc=fnc, memo=memo) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                setattr(obj, k, TensorUtil.apply_deep(v, fnc=fnc, memo=memo))
        return obj
