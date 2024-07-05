from dataclasses import dataclass, field
from awesome.run.multi_runner_config import MultiRunnerConfig
from typing import Dict, Any, List, Literal

@dataclass
class MultiConfigConfig(MultiRunnerConfig):
    """Configuration for the MultiConfig runner. This runner is used to run multiple configs."""

    config_paths: List[str] = field(default_factory=list)
    """Paths to the configs which will be run."""
    
    scan_config_directory: str = field(default="./config")
    """Directory used for config scanning."""

    config_pattern: str = field(default=r".*.yaml")
    """The regex pattern of config files when in scan_dir mode."""

    name_experiment: str = field(default="MultiConfig")


    mode: Literal['plain', 'scan_dir', 'scan_dir_recursive'] = field(default="plain")
    """How the runner should operate. "plain" means that the runner will run the configs in the order they are given in config_paths, "scan_dir" means that the runner will scan the config_directory for configs and run them."""