
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
from typing import Any, Callable, Generator, List, Mapping, Optional, Tuple, Type

import torch
from awesome.event.torch_param_altered_event_args import TorchParamAlteredEventArgs

from awesome.model.dynamic_param_module import DynamicParamModule
from awesome.model.pretrainable_module import PretrainableModule

class AbstractMultiPriorModule(ABC, DynamicParamModule, PretrainableModule):

    priors: torch.nn.ModuleList

    __device__: torch.device

    def __init__(self, 
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.priors = torch.nn.ModuleList()
        self.__device__ = torch.device("cpu")

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.__device__ = device
        return super().to(*args, **kwargs)

    @abstractmethod
    def create_prior(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def create_child_priors(self, num: int = 1):
        """Creates a number of child priors and adds them to the list of priors."""
        if num < 1:
            return
        added_params = dict()
        old_len = len(self.priors)
        
        for i in range(num):
            module = self.create_prior()
            self.priors.append(module)
            added_params.update(dict(module.named_parameters(prefix="priors." + str(old_len + i) + '.')))
        
        # Fire event
        self.param_altered.notify(TorchParamAlteredEventArgs(added_params=added_params))

    def remove_child_priors(self, desired_num: int):
        """Removes a number of child priors from the list of priors.
        The desired number of priors is the number of wanted priors after the removal.
        

        Parameters
        ----------
        desired_num : int
            Number of priors.
        """
        if desired_num == len(self.priors):
            return
        removed_params = dict()
        for i in list(range(desired_num, len(self.priors))[::-1]):
            mod = self.priors.pop(i)
            removed_params.update(dict(mod.named_parameters(prefix="priors." + str(i) + '.')))
        # Fire event
        self.param_altered.notify(TorchParamAlteredEventArgs(removed_params=removed_params))

    def assure_prior_count(self, num: int):
        """Assures that the number of priors is equal to the given number.
        If the number of priors is smaller, the missing priors are created. If the number of priors is larger, the
        additional priors are removed.

        Parameters
        ----------
        num : int
            Number of priors.
        device : torch.device, optional
            Device to which the priors should be moved, by default None.
        """
        if num < len(self.priors):
            self.remove_child_priors(num)
        elif num > len(self.priors):
            self.create_child_priors(num - len(self.priors))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # Check how many priors are within the prior state dict,
        # and create new ones if we don't have enough, and remove if there are too many
        num_inner_priors = len(set([k.split(".")[1] for k in state_dict.keys() if k.startswith('priors.')]))
        self.assure_prior_count(num_inner_priors)
        return super().load_state_dict(state_dict, strict)
    
    def pretrain(self, 
                 *args,
                 **kwargs) -> Any:
        if len(self.priors) == 0:
            self.assure_prior_count(1)
        if not isinstance(self.priors[0], PretrainableModule):
            raise ValueError("Prior module must be a PretrainableModule")
        return self.priors[0].pretrain(*args, **kwargs)
    
    def pretrain_load_state(self, *args, **kwargs) -> None:
        if len(self.priors) == 0:
            self.assure_prior_count(1)
        if not isinstance(self.priors[0], PretrainableModule):
            raise ValueError("Prior module must be a PretrainableModule")
        return self.priors[0].pretrain_load_state(*args, **kwargs)