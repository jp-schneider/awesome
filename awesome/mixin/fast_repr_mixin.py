

from typing import Any, Set
from datetime import datetime, timedelta
import dataclasses
import decimal

ALLOWED_TYPES = set([int, float, str, decimal.Decimal, complex, datetime, timedelta])


class FastReprMixin:
    """Implements a parameterizable repr() function where 
    one can define which properties should be not evaluated as they contain 
    large data slowing performance. Especially useful in interactive environments during debugging."""

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        """Function to ignore fields on the current type during repr.
        Can be overriden.

        Returns
        -------
        Set[str]
            The property names to ignore.
        """
        ignore = []
        if dataclasses.is_dataclass(cls):
            fields = dataclasses.fields(cls)
            for field in fields:
                if field.type not in ALLOWED_TYPES:
                    ignore.append(field.name)
        return set(ignore)

    def __repr__(self) -> str:
        self_dict = dict(vars(self))
        dont_show = type(self).ignore_on_repr()
        for prop in dont_show:
            if prop in self_dict and self_dict.get(prop) is not None:
                self_dict[prop] = "[...]"
        return type(self).__name__ + f"({', '.join([k+'='+repr(v) for k, v in self_dict.items()])})"
