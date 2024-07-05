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

def _encode_buffer(buf: bytes) -> str:
    return base64.b64encode(buf).decode()


def _decode_buffer(buf: str) -> bytes:
    return base64.b64decode(buf.encode())


class TensorValueWrapper(JsonConvertible):

    def __init__(self,
                 value: torch.Tensor = None,
                 decoding: bool = False,
                 max_display_values: int = 100,
                 no_large_data: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.device = str(value.device)
        self.has_data = not no_large_data
        if self.has_data:
            self.data = TensorValueWrapper.to_serialized_string(value)
        else:
            self.data = None
        inner = ""
        if len(value.shape) > 1:
            inner = ", ".join((str(x) for x in value.shape))
        elif len(value.shape) == 1:
            inner = str(value.shape[0]) + ","
        else:
            inner = "1"

        self.shape = "(" + inner + ")"
        self.preview = repr(value)

    @classmethod
    def to_serialized_string(cls, value: torch.Tensor) -> str:
        with io.BytesIO() as buf:
            torch.save(value, buf)
            buf.seek(0)
            return _encode_buffer(buf.read())

    @classmethod
    def from_serialized_string(cls, value: str) -> torch.Tensor:
        with io.BytesIO() as buf:
            buf.write(_decode_buffer(value))
            buf.seek(0)
            return torch.load(buf)

    def to_python(self) -> torch.Tensor:
        if self.has_data:
            return TensorValueWrapper.from_serialized_string(self.data)
        else:
            json_str = self.to_json()
            logging.warning(
                f"Tensor was saved without data, can not recover! Result will be without data. Wrapper value was: {os.linesep + json_str}")
            shp = self.shape.replace("(", "").replace(")", "")
            shp = tuple([int(x) for x in shp.split(",") if len(x) > 0])
            dtype = dynamic_import(self.dtype) if self.dtype.startswith("torch.") else None
            dev = torch.device(self.device) if self.device else torch.device("cpu")
            return torch.zeros(shp).to(dtype=dtype, device=dev)


class JsonTensorSerializationRule(JsonSerializationRule):
    """For Tensors"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.Tensor]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TensorValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TensorValueWrapper(value=value, **kwargs).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TensorValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()
