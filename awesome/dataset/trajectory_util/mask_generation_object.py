
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


from dataclasses import dataclass
import re
import numpy as np


@dataclass
class MaskGenerationObject:
    """This is a small wrapper object to store the information needed to generate a mask from a trajectory file."""

    frame_trajectories: Optional[pd.DataFrame] = None
    """Trajectories of the frame. Rows are Trajectories as the class suggests."""

    img_shape: Optional[Tuple[int, int]] = None
    """Shape of the image in the form (height, width)."""