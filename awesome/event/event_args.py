from dataclasses import dataclass, field
import time
from abc import ABC


@dataclass
class EventArgs(ABC):
    """Base class for all event arguments."""

    cancel: bool = field(default=False)
    """Cancel flag to stop further observer execution."""

    time: float = field(default_factory=time.time)
    """Current time where the event happened."""
