from dataclasses import dataclass, field
from typing import Any, Dict, Set

import torch
from awesome.event import EventArgs
from awesome.mixin import FastReprMixin

@dataclass(repr=False)
class TorchParamAlteredEventArgs(EventArgs, FastReprMixin):
    """Event arguments for modules which change their parameters dynamically."""

    added_params: Dict[str, Any] = field(default_factory=dict)
    """The parameters which were added to the module.
    Should be a dictionary with a key for each parameter name and the value should be the parameter itself. Similar to state_dict"""

    removed_params: Dict[str, Any] = field(default_factory=dict)
    """The parameters which were removed from the module."""

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        """Function to ignore fields on the current type during repr.
        Can be overriden.

        Returns
        -------
        Set[str]
            The property names to ignore.
        """
        return set(['added_params', 'removed_params'])