from typing import Any, Dict, List, Literal, Type
from awesome.error.argument_none_error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
import decimal
from awesome.util.reflection import class_name, dynamic_import

class TypeValueWrapper(JsonConvertible):

    def __init__(self,
                 value: type = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = class_name(value)

    def to_python(self) -> type:
        return dynamic_import(self.value)


class JsonTypeSerializationRule(JsonSerializationRule):
    """For python type instances"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [type]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TypeValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        args = dict(kwargs)
        args['no_uuid'] = True# No uuid for types
        return TypeValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **args)

    def backward(self, value: TypeValueWrapper, **kwargs) -> type:
        return value.to_python()
