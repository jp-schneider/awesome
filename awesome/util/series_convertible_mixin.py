from typing import Any, Dict, List, Optional
import logging
try:
    import pandas as pd
    from pandas import Series, DataFrame
except (NameError, ImportError, ModuleNotFoundError):
    Series = object
    DataFrame = object
    pd = object
    pass  # Ignore exception on import
from awesome.util.format import to_snake_case
from awesome.util.object_factory import ObjectFactory


class SeriesConvertibleMixin:
    """Mixin to indicate that the object can be converted to a series and vice versa."""

    def to_series(self) -> Series:
        """Converts the object to a series.
        Uses by default all entries in __dict__

        Returns
        -------
        Series
            The created series.
        """
        return Series(dict(vars(self)))

    @classmethod
    def from_data_frame(cls, df: DataFrame, allow_dynamic_args: bool = True) -> List[Any]:
        """Generates a list of instances from a dataframe.

        Parameters
        ----------
        df : DataFrame
            The dataframe to create the items from.
        allow_dynamic_args : bool, optional
            If dynamic args should be allowed when creating the object, by default True

        Returns
        -------
        List[Any]
            The list with instances.
        """
        if df is None:
            return []
        return [cls.from_series(x, allow_dynamic_args=allow_dynamic_args) for x in df.iloc]

    @classmethod
    def from_dict(
            cls, _dict: Dict[str, Any],
            snake_case_conversion: bool = False,
            allow_dynamic_args: bool = True,
            additional_data: Optional[Dict[str, Any]] = None) -> Any:
        """Creates the current class from a dict.
        Will also convert keys to snake case and additional data can be inserted.

        Parameters
        ----------
        _dict : Dict[str, Any]
            The dict of entries to create the object from.
        snake_case_conversion : bool, optional
            If keys should be converted to snake case, by default False
        allow_dynamic_args : bool, optional
            If the cls should also accept dynamic arguments, by default True
        additional_data : Optional[Dict[str, Any]], optional
            Additional data to insert in the dict, by default None

        Returns
        -------
        Any
            The created object.
        """
        if additional_data is not None:
            _dict.update(additional_data)
        if snake_case_conversion:
            converted = {}
            for key, value in _dict.items():
                converted[to_snake_case(key)] = value
            _dict = converted
        return ObjectFactory.create_from_kwargs(cls, allow_dynamic_args, ** _dict)

    @classmethod
    def from_series(
            cls, series: Series,
            snake_case_conversion: bool = False,
            allow_dynamic_args: bool = True,
            additional_data: Optional[Dict[str, Any]] = None) -> Any:
        """Create the object from a series. "Inverts" the to_series operation.

        Parameters
        ----------
        series : Series
            The series to create the object from.
        snake_case_conversion : bool, optional
            If names or keys should be converted to snake case, by default False
        allow_dynamic_args : bool, optional
            If dynamic arguments should be allowed, by default True
        additional_data : Optional[Dict[str, Any]], optional
            Additional data to create the object, by default None

        Returns
        -------
        Any
            The created object.
        """
        d = series.to_dict()
        for k, v in d.items():
            is_na = pd.isna(v)
            if isinstance(is_na, bool) and is_na:
                d[k] = None
        return cls.from_dict(d,
                             snake_case_conversion=snake_case_conversion,
                             allow_dynamic_args=allow_dynamic_args,
                             additional_data=additional_data)
