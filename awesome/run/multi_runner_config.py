from dataclasses import dataclass, field
from awesome.run.config import Config
from typing import Optional, Union


@dataclass
class MultiRunnerConfig(Config):
    base_config: Union[str, Config] = None
    """The base config which will be used for the child runners."""
    runner_type: str = None
    """The type of the runner which will be used for the child runners."""
    config_directory: str = field(default="./config")
    """The directory where the child configs will be stored."""
    runner_script_path: str = field(default="./run.py")
    """The path to the runner script."""
    create_job_file: bool = field(default=False)
    """If True, a job file will be created."""
    job_file_path: Optional[str] = None
    """The path to the job file. If None, a job file will be created in the config directory."""
    name_experiment: str = field(default="MultiRunnerConfig")
    """Name for the multi runner, this is usually not used."""
    dry_run: bool = field(default=False)
    """If True, the runner will not execute training."""
    n_parallel: int = field(default=1)
    """Number of parallel executions."""
    preset_output_folder: bool = field(default=False)
    """If True, the output folder for child agents will be preset with a --output-folder argument."""