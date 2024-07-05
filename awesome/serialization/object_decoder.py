import decimal
import logging
from typing import Any, Callable, Dict

from awesome.serialization import object_hook
from awesome.serialization.json_convertible import (JsonConvertible)
from awesome.error import NoIterationTypeError, NoSimpleTypeError


class ObjectDecoder():
    """Converter instance which is capable of recursively decoding an object with a object_hook like in the json library."""

    object_hook: Callable[[Dict[str, Any]], Any]
    """The object hook which is used to decode a dict."""

    @staticmethod
    def default() -> 'ObjectDecoder':
        """Returning the default object decoder with the object_hook.

        Returns
        -------
        ObjectDecoder
            The Object decoder.
        """
        return ObjectDecoder(object_hook=object_hook)

    def __init__(self, object_hook: Callable[[Dict[str, Any]], Any]) -> None:
        self.object_hook = object_hook

    def decode(self, obj: Dict[str, Any]) -> Any:
        """Decodes the object with the given object hook.
        It applies the hook to inner objects first and moves outwards.

        Parameters
        ----------
        obj : Dict[str, Any]
            The object to decode.

        Returns
        -------
        Any
            A new object or class instance.
        """
        # Iterrating threw obj and apply hook on all object structures
        return self._convert_value(obj, "", obj)

    def fix(self, obj: Any) -> Any:
        """Fixes (restores) the internal structure of the object, when some properties
        are dicts rather than the original object.

        Parameters
        ----------
        obj : Any
            The object which can be fixed.

        Returns
        -------
        Any
            The repared object.
        """

    def _convert_value(self, value: Any, name: str, context: Dict[str, Any]) -> Any:
        try:
            return self._convert_simple_type(value, name, context)
        except NoSimpleTypeError:
            try:
                return self._convert_iterable(value, name, context)
            except NoIterationTypeError:
                return value

    def _convert_simple_type(self, value, name: str, object_context: Dict[str, Any]) -> Any:
        if value is None:
            return value
        elif isinstance(value, (int, float, str, decimal.Decimal)):
            return value
        else:
            raise NoSimpleTypeError()

    def _handle_dict(self, value: Dict[str, Any], name: str, object_context: Dict[str, Any]) -> Any:
        # Works same as json hook, creates objects from inside to outside
        ret = {}
        # Handling internals
        for k, v in value.items():
            ret[k] = self._convert_value(v, name, value)
        # Converting ret with hook if childrens are processed
        return self.object_hook(ret)

    def _convert_iterable(self, value, name: str, object_context: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            a = []
            for subval in value:
                a.append(self._convert_value(subval, name, value))
            return a
        elif isinstance(value, (dict)):
            return self._handle_dict(value, name, object_context)
        elif hasattr(value, '__iter__'):
            # Handling iterables which are not lists or tuples => handle them as dict.
            return self._handle_dict(dict(value), name, object_context)
        else:
            raise NoIterationTypeError()
