import base64
import logging
from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from awesome.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
import torch
import io
import numpy as np
import os
from awesome.util.reflection import dynamic_import
from collections import OrderedDict

def _encode_buffer(buf: bytes) -> str:
    return base64.b64encode(buf).decode()


def _decode_buffer(buf: str) -> bytes:
    return base64.b64decode(buf.encode())


class SequentialValueWrapper(JsonConvertible):

    def __init__(self,
                 value: torch.nn.Sequential = None,
                 decoding: bool = False,
                 max_display_values: int = 100,
                 no_large_data: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = dict(value._modules)

    def to_python(self) -> torch.Tensor:
        return torch.nn.Sequential(OrderedDict(self.value))

class JsonSequentialSerializationRule(JsonSerializationRule):
    """For pytorch sequential models."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.nn.Sequential]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [SequentialValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return SequentialValueWrapper(value=value, **kwargs).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: SequentialValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()
