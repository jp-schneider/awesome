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
from torch.nn import ModuleList

class ModuleListValueWrapper(JsonConvertible):

    def __init__(self,
                 value: torch.dtype = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = list(value)

    def to_python(self) -> torch.Tensor:
        return ModuleList(self.value)


class JsonModuleListSerializationRule(JsonSerializationRule):
    """For Torch dtypes"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [ModuleList]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [ModuleListValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return ModuleListValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: ModuleListValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()
