from enum import Enum


class LabelMode(Enum):
    """Label mode for (weak) labels. Defines if the (weak) label is a single object label or a multi object label."""

    SINGLE_OBJECT = "single_object"
    """Defines that the (weak) label is a single object label. So a binary decision between foreground and background."""

    MULTIPLE_OBJECTS = "multiple_objects"
    """Defines that the (weak) label is a multi object label. So a decision between multiple objects and background."""
