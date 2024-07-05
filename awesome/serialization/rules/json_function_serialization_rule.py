import base64
import decimal
import types
from typing import Any, Callable, Dict, List, Literal, Type

from awesome.error.argument_none_error import ArgumentNoneError
from awesome.util.reflection import class_name
from awesome.serialization.json_convertible import JsonConvertible

from .json_serialization_rule import JsonSerializationRule

try:
    import dill
    from dill.source import getsource
except (ImportError, ValueError, ModuleNotFoundError) as err:
    dill = None
    pass
import logging
from inspect import getsource as i_getsource


def encode_callable(fnc: Callable) -> str:
    if dill is None:
        logging.warning(f"Could not serialize a callable as dill is not availible!")
        return None
    dumped = dill.dumps(fnc)
    return base64.b64encode(dumped).decode()


def decode_callable(function_str: str) -> str:
    if function_str is None:
        return None
    decoded = base64.b64decode(function_str.encode())
    return dill.loads(decoded)


def get_callable_source(fnc: Callable) -> str:
    try:
        return getsource(fnc)
    except (Exception) as dill_err:
        try:
            return i_getsource(fnc)
        except Exception as inspect_err:
            logging.warning(
                f"Retrieving sourcecode has failed!\
                Previous exception:\n{dill_err}\n\n")
            return None


class FunctionValueWrapper(JsonConvertible):

    @staticmethod
    def get_source_code_name(name: str):
        return f'__{name}_source__'

    def __init__(self,
                 function: Callable = None,
                 decoding: bool = False,
                 name: str = None,
                 context: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.function = function
        self.source_code = None
        if name is not None and context is not None:
            prop = FunctionValueWrapper.get_source_code_name(name)
            if prop in context:
                self.source_code = context.get(prop)

    def to_json_dict(self, **kwargs) -> Dict[str, Any]:
        as_dict = vars(self)
        fnc = as_dict.pop('function', None)
        if fnc is not None:
            # Serialize fnc with dill and encode as b64 string.
            as_dict['function'] = encode_callable(fnc)
            if self.source_code is not None:
                as_dict['source_code'] = self.source_code
            else:
                as_dict['source_code'] = get_callable_source(fnc)
                if as_dict['source_code'] is None:
                    as_dict['source_code_error'] = "Exception occured, see log for details."

        as_dict['__class__'] = class_name(self)
        return as_dict

    def after_decoding(self):
        if self.function is not None and isinstance(self.function, str):
            self.function = decode_callable(self.function)

    def to_python(self) -> Any:
        return self.function


class JsonFunctionSerializationRule(JsonSerializationRule):
    """For code functions and executables."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [types.FunctionType]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [FunctionValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return FunctionValueWrapper(function=value, name=name, context=object_context).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: FunctionValueWrapper, **kwargs) -> Any:
        return value.to_python()
