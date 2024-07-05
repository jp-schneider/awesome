from typing import Any, Dict, List, Type, Union
import torch.nn as nn
import torch
from awesome.event.torch_param_altered_event_args import TorchParamAlteredEventArgs
from awesome.util.reflection import dynamic_import, class_name
import copy
import inspect
from awesome.model.dynamic_param_module import DynamicParamModule
from awesome.event.event import Event

class MultiPriorModule(DynamicParamModule):
    """Module that can be used to combine multiple prior modules into one. Used when an image 
    should have multiple priors, eg. for multiple objects."""

    def __init__(self, 
                 prior_type: Union[str, Type] = None,
                 prior_args: Dict[str, Any] = None,
                 dtype: torch.dtype = torch.float32,
                 decoding: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if decoding:
            return
        self.priors = nn.ModuleList()
        if prior_args is None:
            prior_args = {}
        if prior_type is None:
            raise ValueError("Prior type must be specified")
        self.prior_type = dynamic_import(prior_type) if isinstance(prior_type, str) else prior_type
        sig = inspect.signature(self.prior_type).parameters
        if 'dtype' not in prior_args and ('dtype' in sig or 'kwargs' in sig):
            prior_args['dtype'] = dtype
        self.prior_args = prior_args if prior_args is not None else {}


    def create_child_priors(self, 
                            num: int = 1, 
                            device: torch.device = None):
        if num < 1:
            return
        added_params = dict()
        old_len = len(self.priors)
        for i in range(num):
            module: torch.nn.Module = self.prior_type(**copy.deepcopy(self.prior_args))
            module.to(device)
            self.priors.append(module)
            added_params.update(dict(module.named_parameters(prefix="priors." + str(old_len + i) + '.')))
        # Fire event
        self.param_altered.notify(TorchParamAlteredEventArgs(added_params=added_params))

    def remove_child_priors(self, desired_num: int):
        if desired_num == len(self.priors):
            return
        removed_params = dict()
        for i in list(range(desired_num, len(self.priors))[::-1]):
            mod = self.priors.pop(i)
            removed_params.update(dict(mod.named_parameters(prefix="priors." + str(i) + '.')))
        # Fire event
        self.param_altered.notify(TorchParamAlteredEventArgs(removed_params=removed_params))

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        # Check if we have a prior for each input
        if len(x) != len(self.priors):
            if len(x) > len(self.priors):
                # Create new priors if we don't have enough
                self.create_child_priors(num = len(x) - len(self.priors), device=x.device)
            elif len(x) < len(self.priors):
                # Remove priors if we have too many
                self.remove_child_priors(len(self.priors) - len(x))
        result = []
        for i in range(len(x)):
            res = self.priors[i](x[i][None, ...], **kwargs)[0] # Remove batch dim
            result.append(res)
        if len(x) == 0:
            return torch.tensor([], device=x.device)
        return torch.stack(result, dim=0) # Add batch dim
    
    def apply_prior(self, prior: Dict[str, Any]) -> None:
        # Check if we have a prior for each input
        # Check how many priors are within the prior state dict, 
        # and create new ones if we don't have enough, and remove if there are too many
        num_inner_priors = len([k for k in prior.keys() if k.startswith('priors.')])
        device = next(list(prior.values())).device if len(prior) > 0 else torch.device("cuda")
        if num_inner_priors > len(self.priors):
            # Create new priors if we don't have enough
            self.create_child_priors(num=num_inner_priors - len(self.priors), device=device)
        elif num_inner_priors < len(self.priors):
            # Remove priors if we have too many
            self.remove_child_priors(num_inner_priors)
        # Load the state dict
        self.load_state_dict(prior)



    def enforce_convexity(self) -> None:
        for prior in self.priors:
            prior.enforce_convexity()