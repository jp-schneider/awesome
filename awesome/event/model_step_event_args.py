
from dataclasses import dataclass
from typing import Any, Dict, Optional

from awesome.agent.util import Tracker
from awesome.event import EventArgs

from awesome.agent.util.learning_mode import LearningMode
from awesome.agent.util.learning_scope import LearningScope


@dataclass
class ModelStepEventArgs(EventArgs):
    """Model event args for a step. Can be in epoch or batch mode."""

    scope: LearningScope = LearningScope.BATCH
    """Defines the scope of the current step, can be epoch or Batch."""

    mode: LearningMode = LearningMode.INFERENCE
    """The learning mode of the model."""

    model: Any = None
    """The model instance which is trained / evaluated by an agent."""

    model_args: Dict[str, Any] = None
    """The init arguments of the model which is trained."""

    input: Any = None
    """The models input data."""

    label: Optional[Any] = None
    """The labels / ground truth / target of the current step."""

    output: Any = None
    """The model prediction or output of the model for input."""

    loss: Any = None
    """The calculated primary loss of the model."""

    loss_name: str = None
    """The name of the actual loss."""

    tracker: Tracker = None
    """The current tracker of the agent, so its model state and loss information."""

    remaining_iterations: Optional[int] = None
    """The number of remaining iterations if predefined."""

    dataset_config: Dict[str, Any] = None
    """The dataset / dataloader configuration."""
