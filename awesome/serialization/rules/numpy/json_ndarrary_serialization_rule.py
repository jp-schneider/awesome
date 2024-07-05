from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from awesome.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
    pass

class NDArrayValueWrapper(JsonConvertible):

    def __init__(self,
                 value: np.ndarray = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.value = value.tolist()

    def to_python(self) -> np.ndarray:
        return np.array(self.value).astype(np.dtype(self.dtype))


class JsonNDArraySerializationRule(JsonSerializationRule):
    """For decimal numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [np.ndarray]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [NDArrayValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return NDArrayValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: NDArrayValueWrapper, **kwargs) -> np.ndarray:
        return value.to_python()
