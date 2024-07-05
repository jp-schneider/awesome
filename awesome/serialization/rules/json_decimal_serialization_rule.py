from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
import decimal


class DecimalValueWrapper(JsonConvertible):

    def __init__(self,
                 value: decimal = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = str(value)

    def to_python(self) -> decimal.Decimal:
        return decimal.Decimal(self.value)


class JsonDecimalSerializationRule(JsonSerializationRule):
    """For decimal numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [decimal.Decimal]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [DecimalValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return DecimalValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: DecimalValueWrapper, **kwargs) -> decimal.Decimal:
        return value.to_python()
