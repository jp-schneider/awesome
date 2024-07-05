from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
from enum import Enum
from awesome.util.reflection import class_name, dynamic_import


class JsonEnumSerializationRule(JsonSerializationRule):
    """For enum types numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [Enum]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return []

    def forward(
            self, value: Enum, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return dict(__class__=class_name(value), value=value.value)

    def backward(self, value: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError()  # Done by others
