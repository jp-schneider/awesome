from collections.abc import Callable
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union
from awesome.serialization import JsonConvertible
try:
    import numpy
except (ModuleNotFoundError, ImportError):
    pass

try:
    import pandas
except (ModuleNotFoundError, ImportError):
    pass

class ObjectEncoder(json.JSONEncoder):
    """Default Encoder to encode common objects to json."""

    def __init__(self, *, 
                 skipkeys: bool = False, 
                 ensure_ascii: bool = True, 
                 check_circular: bool = True, 
                 allow_nan: bool = True, 
                 sort_keys: bool = False, 
                 indent: Optional[Union[int, str]] = None, 
                 separators: Optional[tuple[str, str]] = None, 
                 default: Optional[Callable[..., Any]] = None,
                 json_convertible_kwargs: Optional[Dict[str, Any]] = None
                 ) -> None:
        super().__init__(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular, allow_nan=allow_nan, sort_keys=sort_keys, indent=indent, separators=separators, default=default)
        self.json_convertible_kwargs = json_convertible_kwargs if json_convertible_kwargs is not None else dict()

    def default(self, obj):
        return JsonConvertible.convert_to_json_dict(obj, **self.json_convertible_kwargs)
