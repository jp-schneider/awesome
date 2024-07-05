from dataclasses import dataclass
from typing import Set
import numpy as np
from awesome.mixin.fast_repr_mixin import FastReprMixin

from awesome.util.series_convertible_mixin import SeriesConvertibleMixin

from dataclasses import dataclass, field
import re
import numpy as np


@dataclass(repr=False)
class Trajectory(SeriesConvertibleMixin, FastReprMixin):

    object_id: int = field()
    """Object Id / background id of the tracked object."""

    coordinates: np.ndarray = field()
    """Coordinates of the tracked object in the form (x, y)"""

    frame_ids: np.ndarray = field()
    """Frame ids of the tracked object."""

    line_start: int = field()
    """Start line (1-based) of the trajectory in the tracks file."""

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        return {"coordinates", "frame_ids"}