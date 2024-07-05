from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from awesome.event.model_step_event_args import ModelStepEventArgs

@dataclass
class TorchModelStepEventArgs(ModelStepEventArgs):
    """Specialized step event args for a torch model."""

    model: torch.nn.Module = None
    """The torch model instance used as predictor."""

    optimizer: torch.optim.Optimizer = None
    """The optimizer which is currently used to train the model."""

    output: Tensor = None
    """The model output as tensor."""

    indices: Tensor = None
    """The indices of the dataset which are currently used. Only filled if specified."""
