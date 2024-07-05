from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from awesome.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
    pass

class NumpyGenericWrapper(JsonConvertible):

    def __init__(self,
                 value: np.generic = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.value = value.item()
        if isinstance(self.value, np.generic):
            if isinstance(self.value, (np.longdouble, np.longfloat)):
                self.value = float(self.value)
            else:
                raise ValueError(f"Could not handle dtype: {self.dtype} as numpy cannot unpack it to python type. Specify manually!")


    def to_python(self) -> decimal.Decimal:
        dt = np.dtype(self.dtype)
        return dt.type(self.value)


class JsonGenericSerializationRule(JsonSerializationRule):
    """For decimal numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [np.generic]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [NumpyGenericWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return NumpyGenericWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: NumpyGenericWrapper, **kwargs) -> np.ndarray:
        return value.to_python()
