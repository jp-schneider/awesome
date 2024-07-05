from typing import Dict, Any, List, Tuple, Optional, Set
from enum import Enum
from collections.abc import MutableMapping
from awesome.util.reflection import class_name

class _NOCHANGE:
    pass

class _CYCLE:
    pass

class _MISSING:
    pass

NOCHANGE = _NOCHANGE()
CYCLE = _CYCLE()
MISSING = _MISSING()


def dict_diff(base: Dict[str, Any], cmp: Dict[str, Any]) -> Dict[str, Any]:
    """Returns the difference between two dictionaries.

    Parameters
    ----------
    base : Dict[str, Any]
        The base dictionary.
    cmp : Dict[str, Any]
        The dictionary which will be compared with the base dictionary.

    Returns
    -------
    Dict[str, Any]
        The difference between the two dictionaries.
    """

    result = dict()
    all_keys = set(base.keys()).union(set(cmp.keys()))
    for k in all_keys:
        if k not in base:
            result[k] = cmp[k]
        elif k not in cmp:
            result[k] = MISSING
        else:
            chg = changes(base[k], cmp[k])
            if chg != NOCHANGE:
                result[k] = chg
    if len(result) == 0:
        return NOCHANGE
    return result

def object_diff(base: Any, cmp: Any) -> Dict[str, Any]:
    """Returns the difference between two objects.

    Parameters
    ----------
    base : Any
        The base object.
    cmp : Any
        The object which will be compared with the base object.

    Returns
    -------
    Dict[str, Any]
        The difference between the two objects.
    """

    result = dict()
    if not hasattr(base, "__dict__") or not hasattr(cmp, "__dict__"):
        raise ValueError("Both objects must have a __dict__ attribute.")
    base_dict = base.__dict__
    cmp_dict = cmp.__dict__
    dd = dict_diff(base_dict, cmp_dict)
    if dd != NOCHANGE:
        result.update(dd)
    if type(base) != type(cmp):
        result["__class__"] = class_name(cmp)
    return result

def list_diff(base: List[Any], cmp: List[Any]) -> List[Any]:
    """Returns the difference between two lists.

    Parameters
    ----------
    base : List[Any]
        The base list.
    cmp : List[Any]
        The list which will be compared with the base list.

    Returns
    -------
    List[Any]
        The difference between the two lists.
    """
    result = list()
    length = max(len(base), len(cmp))
    for i in range(length):
        if i >= len(cmp):
            result.append(MISSING)
        elif i >= len(base):
            result.append(cmp[i])
        else:
            result.append(changes(base[i], cmp[i]))
    if all([x == NOCHANGE for x in result]):
        return NOCHANGE
    return result


def tuple_diff(base: tuple, cmp: tuple) -> tuple:
    """Returns the difference between two tuples.

    Parameters
    ----------
    base : tuple
        The base tuple.
    cmp : tuple
        The tuple which will be compared with the base tuple.

    Returns
    -------
    tuple
        The difference between the two tuples.
    """
    result = list_diff(list(base), list(cmp))
    if result == NOCHANGE:
        return NOCHANGE
    return tuple(result)


def changes(base: Any, cmp: Any) -> Any:
    """Returns the changed items of cmp object wrt base.

    Parameters
    ----------
    base : Any
        The base dictionary.
    cmp : Any
        The dictionary which will be compared with the base dictionary.

    Returns
    -------
    Any
        The difference between the two objects. Or NOCHANGE if there are equal
    """

    result = None
    if (base is None and cmp is not None) or (base is not None and cmp is None):
        result = cmp
    if issubclass(type(base), type(cmp)):
        if isinstance(base, dict):
            result = dict_diff(base, cmp)
        elif isinstance(base, Enum):
            if base != cmp:
                result = cmp
            else:
                result = NOCHANGE
        elif isinstance(base, list):
            result = list_diff(base, cmp)
        elif isinstance(base, tuple):
            result = tuple_diff(base, cmp)
        elif hasattr(base, "__dict__") and hasattr(cmp, "__dict__"):
            result = object_diff(base, cmp)
        else:
            if base != cmp:
                result = cmp
            else:
                result = NOCHANGE
    elif isinstance(base, object) and isinstance(cmp, object) and hasattr(base, "__dict__") and hasattr(cmp, "__dict__"):
        result = object_diff(base, cmp)
    else:
        result = cmp
    return result

def flatten(
    dictionary: Dict[str, Any], 
    separator: str = '_', 
    prefix: str='',
    keep_empty: bool = False,
    ) -> Dict[str, Any]:
    """Flattens a dictionary.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to flatten.
    separator : str, optional
        The seperator between nested items, by default '_'
    prefix : str, optional
        Prefix, this is an internal value but can also be used to set a prefix on each entry, by default ''
    keep_empty : bool, optional
        If empty entries should be kept, by default False

    Returns
    -------
    Dict[str, Any]
        The flattend dictionary.
    """
    items = []
    if len(dictionary) == 0 and keep_empty:
        items.append((prefix, None))
    for key, value in dictionary.items():
        new_key = prefix + separator + key if prefix else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, separator=separator, prefix=new_key, keep_empty=keep_empty).items())
        else:
            items.append((new_key, value))
    return dict(items)


def filter(dictionary: Dict[str, Any], allowed_keys: Dict[str, Any]) -> Dict[str, Any]:
    """Filters a dictionary by allowed keys. In A nested way.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to filter.
    allowed_keys : Dict[str, Any]
        The allowed keys.

    Returns
    -------
    Dict[str, Any]
        The filtered dictionary.
    """
    result = dict()
    for key, value in dictionary.items():
        if key in allowed_keys:
            if isinstance(value, MutableMapping):
                # If child is a dict, filter it too but only if there are child keys
                if len(allowed_keys[key]) > 0:
                    result[key] = filter(value, allowed_keys[key])
                else:
                    result[key] = value
            else:
                result[key] = value
    return result

def nested_keys(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Returns all nested keys of a dictionary, whereby the value is a dictionary containing subkeys or is empty if there are no subkeys."""
    result = dict()
    for key, value in dictionary.items():
        if isinstance(value, MutableMapping):
            result[key] = nested_keys(value)
        else:
            result[key] = dict()
    return result

def combine_nested_keys(dictionaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combines nested keys of multiple dictionaries. On all hierarchies of the dict.
    The result contains all keys which exists in at least one dictionary.
    Accordingly its the union of all keys.

    Parameters
    ----------
    dictionaries : List[Dict[str, Any]]
        List of dictionaries with nested keys to combine.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary of all keys.
    """    
    result = dict()
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(value, MutableMapping):
                    result[key] = combine_nested_keys([result[key], value])
                else:
                    result[key] = value
    return result