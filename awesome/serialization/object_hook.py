from typing import Dict, Any, Type, Union, Literal, Callable, Tuple
from datetime import datetime
from enum import Enum
from awesome.util.format import REGEX_ISO_COMPILED
from awesome.util.reflection import dynamic_import
import logging
import inspect
from inspect import Parameter
from awesome.util.reflection import class_name

class _NOTSET:
    pass


NOTSET = _NOTSET()


def _init_object(_type: Type, data: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Custom method to create object when constructing"""
    to_set_data = dict(data)
    to_set_data.pop('__class__', None)

    argspec = list(inspect.signature(_type).parameters)

    # Check if constructor has required args, then these will be passed to the constructor, others will set afterwards.
    init_properties = list(inspect.signature(_type).parameters.items())
    required_args = dict()
    for arg, param in init_properties:
        if param.default == Parameter.empty:
            value = to_set_data.pop(arg, NOTSET)
            equal = (value == NOTSET)
            # If class of value overrides __eq__ method and returns not a bool, then its not NOTSET :) Greetings to numpy and pandas
            if not isinstance(equal, bool) or (not equal):
                required_args[arg] = value
            else:
                # Will raise an error, but this may be intentional
                logging.debug(f"Missing parameter: {arg} for constructing type: {_type.__name__}")

    # Add a flag to the constructor when object is created during decoding.
    if 'kwargs' in argspec or 'decoding' in argspec:
        required_args['decoding'] = True

    try:
        try:
            ret = _type(**required_args)
        except TypeError as err:
            if '__init__()' in err.args[0] and 'decoding' in err.args[0]:
                # Decoding argument should not be provided, remove it and try again.
                required_args.pop('decoding')
                ret = _type(**required_args)
            else:
                raise err
    except TypeError as err:
        if '__init__()' in err.args[0]:
            msg = err.args[0].replace("__init__()", f"__init__() of type: {class_name(_type)}")
            err.args = (msg, ) + err.args[1:]
        raise err
    return ret, to_set_data


def configurable_object_hook(on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> Callable[[Dict[str, Any]], Any]:
    def _object_hook(obj: Dict[str, Any]):
        nonlocal on_error
        from awesome.serialization.rules.json_serialization_rule_registry import JsonSerializationRuleRegistry
        try:
            if '__class__' in obj:
                try:
                    object_type = dynamic_import(obj.get('__class__'))
                except Exception as err:
                    logging.exception(f"Could not import type: {obj.get('__class__')}")
                    raise err
                else:
                    if issubclass(object_type, Enum):
                        return object_type(obj.get("value"))
                    else:
                        # Create object and fill values afterwards
                        obj.pop('__class__')
                        ret = None

                        ret, to_set = _init_object(object_type, obj)

                        for new_name, new_val in to_set.items():
                            setattr(ret, new_name, new_val)

                        if hasattr(ret, 'after_decoding'):
                            fnc = getattr(ret, 'after_decoding')
                            if callable(fnc):
                                fnc()
                            else:
                                raise AttributeError(
                                    f"after_decoding property declared in {object_type.__name__} is not callable!")

                        rule = JsonSerializationRuleRegistry.instance().get_rule_backward(ret)
                        if rule is not None:
                            ret = rule.backward(ret)
                        return ret
            return obj
        except Exception as err:
            if on_error == "raise":
                raise err
            elif on_error == "ignore":
                return obj
            elif on_error == "warning":
                logging.warning(
                    f"{type(err).__name__} in object_hook of obj: {obj['__class__'] if '__class__' in obj else 'No-class'}")
                return obj
            else:
                raise NotImplementedError()
    return _object_hook


object_hook = configurable_object_hook()
"""
Object hook which is used to convert a dictionary to its original python class.
relies on the json convertible class marking logic.

Parameters
----------
obj : Dict[str, Any]
    The object which should be converted.

Returns
-------
Any
    The converted object.

Raises
-------
ImportError, ModuleNotFoundError
    If class is not found / imported

"""


def _check_datetime(value: str) -> Union[str, datetime]:
    match = REGEX_ISO_COMPILED.fullmatch(value)
    if match is not None:
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
    return value
