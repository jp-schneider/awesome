from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
from datetime import datetime

class DatetimeValueWrapper(JsonConvertible):

    def __init__(self,
                 value: datetime = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = value.isoformat()

    def to_python(self) -> complex:
        return datetime.fromisoformat(self.value.replace('Z', '+00:00'))


class JsonDatetimeSerializationRule(JsonSerializationRule):
    """For datetimes"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [datetime]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [DatetimeValueWrapper]

    def forward(
            self, value: datetime, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return DatetimeValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: DatetimeValueWrapper, **kwargs) -> Any:
        return value.to_python()
