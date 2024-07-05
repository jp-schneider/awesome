from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch import Tensor

from awesome.agent.util.tracker import Tracker
from awesome.event.event_args import EventArgs

@dataclass
class TorchTrainingStartedEventArgs(EventArgs):
    """Specialized training started event for a torch model."""

    model: torch.nn.Module = None
    """The torch model instance used as predictor."""

    optimizer: torch.optim.Optimizer = None
    """The optimizer which is currently used to train the model."""

    model_args: Dict[str, Any] = None
    """The init arguments of the model which is trained."""

    loss_name: str = None
    """The name of the actual loss."""

    tracker: Tracker = None
    """The current tracker of the agent, so its model state and loss information."""

    remaining_iterations: Optional[int] = None
    """The number of remaining iterations if predefined."""

    dataset_config: Dict[str, Any] = None
    """The dataset / dataloader configuration."""


