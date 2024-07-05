from typing import Any, Dict, List, Type, Union
import torch.nn as nn
import torch
from awesome.event.torch_param_altered_event_args import TorchParamAlteredEventArgs
from awesome.util.reflection import dynamic_import, class_name
import copy
import inspect
from awesome.event.event import Event


class DynamicParamModule(nn.Module):

    param_altered: Event[TorchParamAlteredEventArgs]
    """Event which fires when the module gets dynamically more parameters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param_altered = Event[TorchParamAlteredEventArgs](source=self)

    