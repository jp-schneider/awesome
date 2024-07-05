from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from awesome.event import EventArgs


@dataclass
class TorchOptimizerCreatedEventArgs(EventArgs):
    """Event arguments for a optimizer created event."""

    optimizer: torch.optim.Optimizer = None
    """The optimizer which was newly created."""

    optimizer_args: Dict[str, Any] = field(default_factory=dict)
    """The init arguments which were applied to create the optimizer"""
