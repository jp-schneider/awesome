import base64
from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from awesome.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
import torch
import io
import numpy as np
from awesome.util.reflection import dynamic_import

class TorchDtypeValueWrapper(JsonConvertible):

    def __init__(self,
                 value: torch.dtype = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = str(value)

    def to_python(self) -> torch.Tensor:
        return dynamic_import(self.value)


class JsonTorchDtypeSerializationRule(JsonSerializationRule):
    """For Torch dtypes"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.dtype]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TorchDtypeValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TorchDtypeValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TorchDtypeValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()
