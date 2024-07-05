from typing import Any, Dict, List, Literal, Optional, Type
from uuid import UUID
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
from awesome.serialization.json_convertible import convert

class SetValueWrapper(JsonConvertible):

    def __init__(self,
                 value: list = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.values = value

    def to_python(self) -> complex:
        return set(self.values)


class JsonSetSerializationRule(JsonSerializationRule):
    """For python sets"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [set]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [SetValueWrapper]

    def forward(
            self, value: set, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = set()
        return SetValueWrapper(list(value)).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

    def backward(self, value: SetValueWrapper, **kwargs) -> Any:
        return value.to_python()
