
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
import types
from typing import Any, Callable, Generator, List, Mapping, Optional, Tuple, Type, Union

import torch
from awesome.event.torch_param_altered_event_args import TorchParamAlteredEventArgs

from awesome.model.abstract_multi_prior_module import AbstractMultiPriorModule
from awesome.model.dynamic_param_module import DynamicParamModule


class NumberBasedMultiPriorModule(AbstractMultiPriorModule):
    """Multi prior module which contains a prior per given count."""

    def __init__(self, 
                 prior: torch.nn.Module = None,
                 prior_type: Optional[Type[torch.nn.Module]] = None, 
                 prior_args: dict = None,
                 min_priors: int = 1,
                 ) -> None:
        super().__init__()
        if prior is not None:
            prior.to(device=torch.device("cpu"))
        self.native_prior = prior
        self.prior_type = prior_type if prior_type is not None else type(prior)
        self.prior_args = prior_args if prior_args is not None else {}
        # Assure that we have at least min_priors within the model
        self.assure_prior_count(min_priors)

    def create_prior(self, *args, **kwargs):
        prior = None
        if self.native_prior is not None:
            prior = copy.deepcopy(self.native_prior)
        else:
            prior = self.prior_type(**copy.deepcopy(self.prior_args))
        prior.to(device=self.__device__)
        return prior
    
    def _wrapped(self, fnc: Callable[[Any, Mapping[str, Any]], torch.Tensor], *args, num_priors: int, **kwargs) -> torch.Tensor:
        # Check if we have a prior for each input
        self.assure_prior_count(num_priors)
        result = []
        for i in range(num_priors):
            res = fnc(self.priors[i], *args, **kwargs)
            result.append(res)
        return torch.stack(result, dim=1) # Stack on channel dim

    def forward(self, *args, num_priors: int, **kwargs) -> torch.Tensor:
        fwd = self.prior_type.forward
        return self._wrapped(fwd, *args, num_priors=num_priors, **kwargs)

            
    @classmethod
    def get_type_args(cls, inner_factory: Union[Type[torch.nn.Module], types.FunctionType], inner_args, **kwargs) -> Tuple[Union[Type[torch.nn.Module], types.FunctionType], dict]:
        """Method for nesting the creation of multi prior modules within each other.

        Parameters
        ----------
        inner_factory : Union[Type[torch.nn.Module], types.FunctionType]
            Inner factory function or class for constructing the inner / child prior.

        inner_args : _type_
            Inner arguments for the inner factory.

        Returns
        -------
        Tuple[Union[Type[torch.nn.Module], types.FunctionType], dict]
            The type / factory and the arguments for the type / factory.
        """
        outer_args = dict(prior_type=inner_factory, prior_args=inner_args, **kwargs)
        return (cls, outer_args)