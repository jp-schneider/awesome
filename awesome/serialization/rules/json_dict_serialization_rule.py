from enum import Enum
import logging
from typing import Any, Dict, List, Literal, Optional, Type

from awesome.serialization.json_convertible import convert

from .json_serialization_rule import JsonSerializationRule
from uuid import UUID
import inspect
import torch

ALLOWED_KEY_TYPES = [str, int, float, bool, None]

class JsonDictSerializationRule(JsonSerializationRule):
    """For lists of objects."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [dict]

    def is_forward_applicable(self, value: Any) -> bool:
        res = super().is_forward_applicable(value)
        if res:
            if not all([isinstance(k, tuple(ALLOWED_KEY_TYPES)) for k in value.keys()]):
                logging.warning(f"Keys of dict are not of allowed types: {value.keys()}")
                return False
            return res
        return not (isinstance(value, type)) and (hasattr(value, '__iter__') or hasattr(value, 'to_json_dict'))

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return []

    def forward(
            self, 
            value: Any, 
            name: str, 
            object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = dict()
        if fnc := getattr(value, "to_json_dict", None):
            args = dict()
            sig = inspect.signature(fnc)
            if "memo" in sig.parameters:
                args["memo"] = memo
            if "handle_unmatched" in sig.parameters:
                args["handle_unmatched"] = handle_unmatched
            if "kwargs" in sig.parameters:
                args.update(kwargs)
            if callable(fnc):
                return fnc(**args)
            raise ValueError("to_json_dict is not callable!")
        elif hasattr(value, '__iter__'):
            # Handling iterables which are not lists or tuples => return them as dict.
            # Iterate over items
            ret = {}
            as_dict = dict(value)
            for k, v in as_dict.items():
                ret[k] = convert(v, k, as_dict, handle_unmatched=handle_unmatched, memo=memo, **kwargs)
            return ret
        else:
            raise ValueError(f"Dont know how to handle Type: {type(value)}")

    def backward(self, value: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError()  # Done by others
