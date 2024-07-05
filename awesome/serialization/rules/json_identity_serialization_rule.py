from typing import Any, Dict, List, Literal, Type
from .json_serialization_rule import JsonSerializationRule


class JsonIdentitySerializationRule(JsonSerializationRule):
    """Simple identity rule for objects which can be mapped to json directly."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [int, str, float]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [int, str, float]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return value

    def backward(self, value: Any, **kwargs) -> Any:
        return value
