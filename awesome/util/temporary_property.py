from typing import Any, Dict
from awesome.error import ArgumentNoneError
import logging


class _NOTSET():
    pass


class _PATHNONE():
    pass


NOTSET = _NOTSET()
"""Constant for a non existing value."""

PATHNONE = _PATHNONE()
"""Constant for a non existing path."""


def _get_nested_value(obj: Any, path: str, default: Any = NOTSET) -> Any:
    if obj is None and len(path) > 0:
        return PATHNONE
    if '.' in path:
        path, rest = path.split('.', 1)
        return _get_nested_value(getattr(obj, path, default), rest, default)
    else:
        return getattr(obj, path, default)


def _set_nested_value(obj: Any, path: str, value: Any):
    if value == PATHNONE:
        return
    if '.' in path:
        path, rest = path.split('.', 1)
        return _set_nested_value(getattr(obj, path), rest, value)
    else:
        if value == NOTSET:
            delattr(obj, path)
        else:
            setattr(obj, path, value)


class TemporaryProperty():
    """Context manager for altering a object temporary and set properties back to their original value."""

    def __init__(self, obj: Any, nested_values: Dict[str, Any] = None, **kwargs):
        """Creates the context manager by an object and kwargs defining the 
        properties to alter with their value.

        Parameters
        ----------
        obj : Any
            The object which should be altered.

        nested_values : Dict[str, Any], optional
            A dictionary for nested values to alter.
            Keys can be paths to nested properties, e.g. "a.b.c".
            Values are the new values for the properties.

        kwargs:
            Property names and values to alter.

        Raises
        ------
        ArgumentNoneError
            If obj is None.
        """
        if obj is None:
            raise ArgumentNoneError("obj")
        if nested_values is None:
            nested_values = dict()
        self.obj = obj
        self.old_values = dict()
        args = dict(kwargs)
        args.update(nested_values)
        self.intermediate_values = args

    def __enter__(self):
        for k, v in self.intermediate_values.items():
            self.old_values[k] = _get_nested_value(self.obj, k, NOTSET)
            if self.old_values[k] == NOTSET:
                logging.warning(f"Property: {k} was not existing in: {repr(self.obj)}!")
            if self.old_values[k] != PATHNONE:
                _set_nested_value(self.obj, k, v)       

    def __exit__(self, type, value, traceback):
        for k, v in self.old_values.items():
            _set_nested_value(self.obj, k, v)
        return False
