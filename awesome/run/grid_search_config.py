from dataclasses import dataclass, field
from awesome.run.config import Config
from awesome.run.multi_runner_config import MultiRunnerConfig
from typing import Dict, Any, List, Union

@dataclass
class GridSearchConfig(MultiRunnerConfig):

    base_config: Union[str, Config, List[Config]] = None
    """The base config which will be used for the child runners. If it is a list, the cartesian product will be created for each config in the list."""

    param_grid: Dict[str, Any] = field(default_factory=dict)
    """The parameter grid which will be used to create the child configs."""
    
    name_experiment: str = field(default="GridSearch")