from dataclasses import dataclass
from typing import Optional

from awesome.serialization.json_convertible import JsonConvertible

from dataclasses import dataclass, field
import re
import numpy as np

@dataclass
class FrameDescription(JsonConvertible):

    TRAJECTORY_FILE_PARSING_PATTERN = r"(Frame number:)(\r)?\n(?P<frame_number>[\d]+)(\r)?\n(File name:)(\r)?\n(?P<ground_truth_file_name>[\w\.\-_\d]+)(\r)?\n(Input file name:)(\r)?\n(?P<image_file_name>[\w\.\-_\d]+)(\r)?\n"

    IMAGE_NAME_PATTERN = r"^(?P<dataset_name>[A-z\-]+)((?P<numbering>[0-9]+)_)?(?P<frame_name>[\d]+)\.((png)|(jpg)|(ppm)|(pgm))$"

    frame_number: int = field(default=None)
    """Frame number as integer which corresponds to the number in the trajectory file."""

    frame_name: int = field(default=None)
    """Frame name / number as integer which corresponds to the number in the image file name."""

    ground_truth_file_name: Optional[str] = field(default=None)

    image_file_name: str = field(default=None)
