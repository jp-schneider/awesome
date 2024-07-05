from typing import Any, Dict, List, Literal, Optional, Type
from uuid import UUID

from awesome.serialization.json_convertible import JsonConvertible, convert

from .json_serialization_rule import JsonSerializationRule


class TupleValueWrapper(JsonConvertible):

    def __init__(self,
                 value: list = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = value

    def to_python(self) -> tuple:
        return tuple(self.value)


class JsonTupleSerializationRule(JsonSerializationRule):
    """For Tuple of objects."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [tuple]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TupleValueWrapper]

    def forward(
            self, 
            value: tuple, 
            name: str, 
            object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = set()
        return TupleValueWrapper(list(value)).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

    def backward(self, value: TupleValueWrapper, **kwargs) -> Any:
        return value.to_python()
