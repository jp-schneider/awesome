from typing import Any, Dict, List, Literal, Type
from awesome.serialization.json_convertible import JsonConvertible
from awesome.serialization.rules.json_serialization_rule import JsonSerializationRule
import torch

class TorchDeviceValueWrapper(JsonConvertible):

    def __init__(self,
                 value: torch.device = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = str(value)

    def to_python(self) -> torch.Tensor:
        return torch.device(self.value)


class JsonTorchDeviceSerializationRule(JsonSerializationRule):
    """For Torch devices"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.device]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TorchDeviceValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TorchDeviceValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TorchDeviceValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()
