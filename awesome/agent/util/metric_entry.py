from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
from decimal import Decimal
from awesome.util.series_convertible_mixin import SeriesConvertibleMixin
from awesome.util.reflection import class_name, dynamic_import
import inspect


@dataclass
class MetricEntry(SeriesConvertibleMixin):
    """Defines a metric entry."""

    tag: str = field(default=None)
    """Tag for the metric. Should something which describes what the metric is."""

    value: Union[float, int, complex, Decimal] = field(default=None)
    """The actual value for the metric."""

    step: int = field(default=-1)
    """The step value, when the metric occured."""

    global_step: int = field(default=-1)
    """A global step value which counts total progress."""

    metric_qualname: str = field(default=None)
    """The fully qualifying name of the metric class / loss function. Will be used to compare metrics."""

    @staticmethod
    def df_fields() -> List[str]:
        """Returning the fields which will be stored in a dataframe when using metric summary."""
        return ["value", "step", "global_step"]

    def _get_loss(self) -> Optional[Any]:
        """Get the loss object or function based on its qualname.

        Returns
        -------
        Optional[Any]
            The executable loss or None.
        """
        if self.metric_qualname is None:
            return None
        imported = dynamic_import(self.metric_qualname)
        if inspect.isclass(imported):
            return imported()  # Create instance of the class
        else:
            return imported
