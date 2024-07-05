from typing import Any, Type, Union
from types import ModuleType, FunctionType
import importlib

IMPORT_CACHE = {}


def dynamic_import(class_or_method: str) -> Any:
    """
    Imports a class, method or module based on a full import string.
    Example: For importing this method the string would be:

    `awesome.util.reflection.dynamic_import`
    So this method dynamically does the following:
    >>> from awesome.util.reflection import dynamic_import

    Also modules can be imported with a string like:

    `awesome`

    Meaning:

    >>> import awesome

    Parameters
    ----------
    class_or_method : str
        The fully qualifing string to import. 
        Working strings can be retrieved with ``class_name`` function.

    Returns
    -------
    Any
        The imported module / type.

    Raises
    ------
    ImportError
        If the import fails.
    """
    value = IMPORT_CACHE.get(class_or_method, None)
    if value is not None:
        return value
    components = class_or_method.split('.')
    if len(components) > 1:
        # Class / type import, trim the class
        module = components[:-1]
    else:
        module = components
    try:
        mod = importlib.import_module(".".join(module))
    except (NameError, ModuleNotFoundError, ImportError) as err:
        raise ImportError(f"Could not import: {class_or_method} \
                          due to an {err.__class__.__name__} does \
                          the Module / Type exists and is it installed?") from err
    if len(components) == 1:
        # Import was a module import only, return it directly
        return mod
    attribute = getattr(mod, components[-1])
    IMPORT_CACHE[class_or_method] = attribute
    return attribute


def class_name(cls_or_obj: Union[object, Type]) -> str:
    """
    Returns the class name of the current class or object as string with namespace.

    Parameters
    ----------
    cls_or_obj : Union[object, Type]
        Class or object to get its fully qualified name.

    Returns
    -------
    str
        The fully qualified name.
    """
    if isinstance(cls_or_obj, (Type, FunctionType)): # Types and functions can be imported via their name
        return cls_or_obj.__module__ + '.' + cls_or_obj.__name__
    return cls_or_obj.__class__.__module__ + '.' + cls_or_obj.__class__.__name__
