from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type
from uuid import UUID

from awesome.serialization.json_convertible import convert

from .json_serialization_rule import JsonSerializationRule


class JsonListSerializationRule(JsonSerializationRule):
    """For lists of objects."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [list]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return []

    def forward(
            self, value: list, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = set()
        a = []
        for subval in value:
            a.append(convert(subval, name, object_context, handle_unmatched=handle_unmatched, memo=memo, **kwargs))
        return a

    def backward(self, value: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError()  # Done by others
