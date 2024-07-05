import logging
import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, TypeVar, Union, get_type_hints

from awesome.agent.agent import Agent
from awesome.agent.util import MetricEntry
from awesome.util.format import to_snake_case
from awesome.util.path_tools import numerated_file_name

from awesome.dataset import BaseDataset
from awesome.run.config import Config
from awesome.error import ArgumentNoneError
from awesome.util.reflection import class_name
import random
import torch
import numpy as np

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Runner():
    """Runner which is used to run the experiment."""

    agent: Agent
    """Agent which is used to run the experiment."""

    dataloader: BaseDataset
    """Data loader which is used to train the agent."""

    config: Config
    """Configuration of the runner."""

    diff_config: Optional[Dict[str, Any]]
    """Dictionary which contains the difference between the base config and the child config of the runner, 
    if the runner is part of a multi runner."""

    parent: Optional['Runner']
    """Parent runner if the runner is part of a multi runner."""

    __runner_context__: Dict[str, Any]
    """Context vars which are additionally
     declared within the agent. 
     These vars are not annotated with type hints and values are autmatically filled within if assigned to instance."""

    __hints__: Dict[Type, Dict[str, Type]] = dict()

    __saved_config__: Optional[str]
    """String config which was saved."""

    @classmethod
    def _is_type_hinted_var(cls, name: str) -> bool:
        if cls not in cls.__hints__:
            cls.__hints__[cls] = get_type_hints(cls)
        hints = cls.__hints__[cls]
        return name in hints

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(**kwargs)
        if config is None:
            raise ArgumentNoneError("config")
        self.config = config
        self.parent = None
        self.diff_config = None
        self.config.used_runner_type = class_name(self)
        self.__runner_context__ = dict()
        self.__saved_config__ = None
        self.set_seed()

    def set_seed(self) -> None:
        """Sets the seed of the random number generators.
        This is used to make the experiments reproducible.
        """        
        seed_all(self.config.seed)

    def store_config(self, **kwargs) -> str:
        """Stores the config of the runner to a file in the agent folder.

        Returns
        -------
        str
            The path where the config is stored.
        """
        path = os.path.join(
            self.agent.agent_folder,
            f"init_cfg_{to_snake_case(type(self.config).__name__)}.yaml")
        path = numerated_file_name(path)
        path = self.config.save_to_file(path, no_uuid=True, no_large_data=True)
        with open(path, 'r') as f:
            self.__saved_config__ = f.read()
        return path

    def log_config(self) -> None:
        """Logs the config of the runner.
        """
        if self.__saved_config__ is None:
            self.store_config()
        logging.info(f"Using Config:\n{self.__saved_config__}")
        
    @abstractmethod
    def patch_agent(self, agent: Agent) -> None:
        """Sets a given agent to the runner by patching in the agent all necessary context
        veriables.

        Parameters
        ----------
        agent : Agent
            The restored agent.
        """        
        pass

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Method to build the runner and makes all necessary preparations.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

    def score(self, *args, **kwargs) -> float:
        """Returns the score of an agent which is the primary metric of the agent
        this is used to compare agents.

        Returns
        -------
        float
            The score of the agent.
        """
        # Returns the score of an agent which is the primary metric of the agent
        score: MetricEntry = self.tracker.get_recent_performance()
        return score.value

    def __setattribute__(self, name: str, value: Any) -> None:
        if type(self)._is_type_hinted_var(name):
            object.__setattribute__(self, name, value)
        else:
            self.__runner_context__[name] = value

    def __getattr__(self, name: str) -> Any:
        if type(self)._is_type_hinted_var(name):
            return object.__getattribute__(self, name)
        else:
            return self.__runner_context__[name]
