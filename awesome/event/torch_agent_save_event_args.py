from dataclasses import dataclass
from typing import Any, Dict


from awesome.event.agent_save_event_args import AgentSaveEventArgs


@dataclass
class TorchAgentSaveEventArgs(AgentSaveEventArgs):
    """Special Agent save event args for the torch agent."""

    optimizer_type: type = None
    """The class of the optimizer."""

    optimizer_args: Dict[str, Any] = None
    """Initial arguments for the optimizer."""
