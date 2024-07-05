import json
from typing import Any, Dict, List, Literal, Type

import jsonpickle

from awesome.util.reflection import class_name
from awesome.serialization.json_convertible import JsonConvertible

from .json_serialization_rule import JsonSerializationRule


class JsonPickleValueWrapper(JsonConvertible):
    """Can wrap any object by pickling it."""

    def __init__(self, obj: object = None, **kwargs):
        if 'decoding' in kwargs:
            return
        self.object = obj

    def to_json_dict(self, **kwargs) -> Dict[str, Any]:
        as_dict = vars(self)
        obj = as_dict.pop('object', None)
        if obj is not None:
            # Serialize the object
            ser = jsonpickle.dumps(obj)
            obj_dict = json.loads(ser)
            as_dict['object'] = obj_dict

        as_dict['__class__'] = class_name(self)
        return as_dict

    def after_decoding(self):
        if self.object is not None and isinstance(self.object, dict):
            ser = json.dumps(self.object)
            self.object = jsonpickle.loads(ser)

    def to_python(self) -> Any:
        return self.object


class JsonPickleSerializationRule(JsonSerializationRule):
    """For Objects in general when no other open is available."""

    def __init__(self, priority: int = 1000) -> None:
        super().__init__(priority=priority)

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return []

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [JsonPickleValueWrapper]

    def forward(
            self, value: tuple, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return JsonPickleValueWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: JsonPickleValueWrapper, **kwargs) -> Any:
        return value.to_python()