import argparse
import inspect
import logging
from argparse import ArgumentParser
from dataclasses import MISSING, Field
from typing import Any, Dict, List, Optional, Type, get_args, ClassVar

from simple_parsing.docstring import get_attribute_docstring
from typing_inspect import is_literal_type, is_optional_type, is_tuple_type, is_classvar

from awesome.error import UnsupportedTypeError, IgnoreTypeError
from enum import Enum

WARNING_ON_UNSUPPORTED_TYPE = True
"""If true, a warning will be printed if a type is not supported."""

def set_warning_on_unsupported_type(warning: bool) -> None:
    """Sets the warning on unsupported type.

    Parameters
    ----------
    warning : bool
        If true, a warning will be printed if a type is not supported.
    """
    global WARNING_ON_UNSUPPORTED_TYPE
    WARNING_ON_UNSUPPORTED_TYPE = warning

class ParseEnumAction(argparse.Action):
    """Custom action to parse enum values from the command line."""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        self.enum_type = kwargs.pop("enum_type", None)
        super().__init__(option_strings, dest, **kwargs)
        if self.enum_type is None:
            raise ValueError("enum_type must be specified")

    def __call__(self, parser, namespace, values, option_string=None):
        v = None
        if isinstance(values, str):
            v = self.enum_type(values)
        elif isinstance(values, list):
            v = [self.enum_type(x) for x in values]
        else:
            raise ValueError(f"Unsupported type of values: {type(values).__name__}")
        setattr(namespace, self.dest, v)


class ArgparserMixin:
    """Mixin wich provides functionality to construct a argparser for a 
    dataclass type and applies its data."""

    @classmethod
    def _map_type_to_parser_arg(cls, field: Field, _type: Optional[Type] = None) -> Dict[str, Any]:
        """Mapping field types to argparse arguments.
        Parameters
        ----------
        field : Field
            The field which should be mapped.
        _type : Optional[Type]
            Alterating field type on recursive calls, default None.

        Returns
        -------
        Dict[str, Any]
            kwargs for the argparse add argument call.
        Raises
        ------
        UnsupportedTypeError
            If the type is not supported for comparison.
        """
        if not _type:
            _type = field.type
        if isinstance(_type, Type) and issubclass(_type, bool):
            # Check default and switch accordingly
            if not field.default:
                return dict(action="store_true")
            else:
                return dict(action="store_false")
        elif isinstance(_type, Type) and issubclass(_type, (str, int, float)):
            return dict(type=_type)
        elif is_literal_type(_type):
            arg = get_args(_type)
            ret = dict()
            if len(arg) > 0:
                ret["choices"] = list(arg)
                ret["type"] = type(arg[0])
            return ret
        elif ((isinstance(_type, Type) and
              (issubclass(_type, list) or issubclass(_type, tuple)))
              or is_tuple_type(_type)):
            # Handling list or tuples the same way
            # Limitation: Lists can have an arbitrary amount of arguments.

            args = dict()
            arg = get_args(_type)
            if is_tuple_type(_type):
                # If a typing tuple, the number types can be directly inferred
                args["nargs"] = len(arg)
            elif issubclass(_type, tuple):
                # Empty tuple would not make sense so limit it to 1...n
                args["nargs"] = "+"
            else:
                # For list empty would be ok.
                args["nargs"] = "*"
            if len(arg) > 0:
                args["type"] = arg[0]
            return args
        elif isinstance(_type, Type) and issubclass(_type, Enum):
            choices = [x.value for x in _type]
            # Get type of choice
            _arg_type = type(next((x for x in choices), str))
            return dict(type=_arg_type,
                        enum_type=_type,
                        choices=choices,
                        action=ParseEnumAction)
        elif is_optional_type(_type):
            # Unpack optional type.
            _new_type = get_args(_type)[0]
            args = cls._map_type_to_parser_arg(field, _new_type)
            # Because its optional, make it non required
            args["required"] = False
            return args
        elif is_classvar(_type):
            raise IgnoreTypeError()
        else:
            raise UnsupportedTypeError(
                f"Dont know how to handle type: {_type} of field: {field.name}.")

    @classmethod
    def _get_parser_arg_value(cls, field: Field, value: Any, _type: Optional[Type] = None) -> Any:
        if not _type:
            _type = field.type
        if (isinstance(_type, Type) and issubclass(_type, (str, int, float, bool))) or value is None:
            return value  # Simple types
        elif is_literal_type(_type):
            arg = get_args(_type)
            ret = dict()
            if len(arg) > 0:
                if value not in arg:
                    raise ValueError(f"{value} is not value supported for literal type: {_type}")
                return value
            else:
                raise ValueError(f"Could not specify {value} for literal type: {_type}")
        elif ((isinstance(_type, Type) and
              (issubclass(_type, list) or issubclass(_type, tuple)))
              or is_tuple_type(_type)):
            if is_tuple_type(_type) or issubclass(_type, tuple):
                return tuple(value)  # Be shure that value is a tuple
            else:
                # For list empty would be ok.
                return list(value)
        elif is_optional_type(_type):
            # Unpack optional type.
            _new_type = get_args(_type)[0]
            return cls._get_parser_arg_value(field, value=value, _type=_new_type)
        elif isinstance(_type, Type) and issubclass(_type, Enum):
            return value
        else:
            raise UnsupportedTypeError(
                f"Dont know how to handle type: {_type} of field: {field.name}.")

    @classmethod
    def _get_parser_members(cls) -> List[Field]:
        """Returning the parser members which are in the dataclass.
        """
        # Get all dataclass properties
        members = inspect.getmembers(cls)
        all_fields: List[Field] = list(
            next((x[1] for x in members if x[0] == '__dataclass_fields__'), dict()).values())

        # Non private fields
        fields = [x for x in all_fields if not x.name.startswith('_') and x.name not in cls.argparser_ignore_fields()]
        return fields

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        """Can be derived to ignore custom fields and not apply them in the argparser.

        Returns
        -------
        List[str]
            List of fields to ignore.
        """
        return [

        ]

    @classmethod
    def get_parser(cls, parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        """Creates / fills an Argumentparser with the fields of the current class.
        Inheriting class must be a dataclass to get annotations and fields.
        By default only puplic field are used (=field with a leading underscore "_" are ignored.)
        Parameters
        ----------
        parser : Optional[ArgumentParser]
            An existing argument parser. If not specified a new one will be created. Defaults to None.
        Returns
        -------
        ArgumentParser
            The filled argument parser.
        """
        # Create parser if None
        if not parser:
            parser = argparse.ArgumentParser(
                description=f'Default argument parser for {cls.__name__}',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        fields = cls._get_parser_members()

        for field in fields:
            name = field.name.replace("_", "-")
            try:
                args = cls._map_type_to_parser_arg(field)
            except IgnoreTypeError as ig:
                continue
            except UnsupportedTypeError as err:
                msg = f"Could not create parser arg for field: {field.name} due to a {type(err).__name__} \n {str(err)}"
                if WARNING_ON_UNSUPPORTED_TYPE:
                    logging.warning(msg)
                else:
                    logging.debug(msg)
                continue
            # default value
            if field.default != MISSING:
                args["default"] = field.default
            # docstring
            args["help"] = str(get_attribute_docstring(
                cls, field_name=field.name).docstring_below)
            parser.add_argument("--" + name, **args)

        return parser

    @classmethod
    def from_parsed_args(cls, parsed_args: Any) -> 'ArgparserMixin':
        """Creates an ArgparserMixin object from parsed_args which is the result
        of the argparser.parse_args() method.
        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.
        Returns
        -------
        ArgparserMixin
            The instance of the object filled with cli data.
        """
        fields = cls._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            if hasattr(parsed_args, field.name):
                value = getattr(parsed_args, field.name)
                ret[field.name] = cls._get_parser_arg_value(field, value)
        return cls(**ret)

    def apply_parsed_args(self, parsed_args: Any) -> None:
        """Applies parsed_args, which is the result
        of the argparser.parse_args() method, to an existing object.
        But only if its different to the objects default.
        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.
        """
        fields = type(self)._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            if hasattr(parsed_args, field.name):
                value = getattr(parsed_args, field.name)
                if ((field.default != MISSING and (value is not None and value != field.default)) or
                        (field.default_factory != MISSING and (value is not None and value != field.default_factory()))
                        or (field.default == MISSING and field.default_factory == MISSING)):
                    setattr(self, field.name, self._get_parser_arg_value(field, value))
