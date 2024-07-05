
from dataclasses import dataclass
from typing import Any, Dict, Optional

from awesome.event import EventArgs


@dataclass
class TrainingFinishedEventArgs(EventArgs):
    """Event args for when the training was done completely."""

    training_error_occurred: Optional[Exception] = None
    """If an error occurred during training, this will be set to the exception."""

    