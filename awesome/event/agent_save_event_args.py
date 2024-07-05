from enum import Enum
import io
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional

from awesome.agent.util import Tracker
from awesome.event import EventArgs
 

class SaveStage(Enum):
    
    UNKNOWN = "unknown"
    """Unknown stage, should not be used."""
    PRETRAINING = "pretraining"
    """Pretraining stage, before the actual training."""
    BEST = "best"
    """Best current model, occurs during training, usally multiple times."""
    END = "end"
    """End of training, after the last epoch."""

@dataclass
class AgentSaveEventArgs(EventArgs):
    """Event arguments for agent save."""

    name: str = None
    """The Name of the agent."""

    tracker: Tracker = None
    """Config for the current agent model state."""

    model_type: type = None
    """The model class for the agent."""

    model_args: Dict[str, Any] = None
    """The arguments which are needed to init the model class."""

    agent_checkpoint: Any = None
    """The current state of the agent, so its checkpoint."""

    dataset_config: Dict[str, Any] = None
    """Dataset configuration for the training dataset."""

    execution_context: Dict[str, Any] = field(default_factory=dict)
    """Parameters which are in the execution context, like kwargs."""
    
    is_training_done: bool = False
    """ Parameter to know if training is done. To save stuff just on last epoch"""

    occurred_training_error: Optional[Exception] = None
    """If an error occurred during training, this will be set to the exception. 
    If this is the case, the event was invoked as the training will stop because of an error."""

    stage: SaveStage = SaveStage.UNKNOWN
    """The current stage of the save."""
