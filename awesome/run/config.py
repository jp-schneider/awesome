import os
from dataclasses import dataclass, field

from awesome.mixin import ArgparserMixin
from awesome.serialization import JsonConvertible
from awesome.util.diff import changes, NOCHANGE
from typing import Any, Dict
import logging


@dataclass
class Config(JsonConvertible, ArgparserMixin):
    """Basic config for a runner."""

    name_experiment: str = field(default="Test")
    """Name of the experiment / agent. Agent will create a subdirectory for each experiment. Default is "Test"."""

    runs_path: str = field(default=os.path.abspath("./runs/"))
    """Base directory where the runs are stored. Agent will create a subdirectory for each run. Default is ./runs/."""

    output_folder: str = field(default=None)
    """Folder where all outputs are stored overrides runs_path. Default is None."""

    diff_config: str = field(default=None)
    """When config is altered from another, this can be used to propagate diff values."""

    use_progress_bar: bool = field(default=True)
    """If a progressbar should be used."""

    run_script_path: str = field(default=None)
    """Path to the run script. Saves the executable path of the script where the run was started with."""

    used_runner_type: str = field(default=None)
    """Type of the runner which was used to perform the experiment."""

    seed: int = field(default=42)
    """Seed for the initialization of the random number generator. Before large random operations in synthetic setting model."""


    def __post_init__(self):
        if self.diff_config is not None and isinstance(self.diff_config, str):
            if os.path.exists(self.diff_config):
                self.diff_config = JsonConvertible.load_from_file(self.diff_config)
            else:
                self.diff_config = JsonConvertible.from_json(self.diff_config)

    def compute_diff(self, other: 'Config') -> Dict[str, Any]:
        """Computes the differences of the current object to another.
        Result will be the changed properties from self to other.

        Parameters
        ----------
        other : Config
            The object to compare with.

        Returns
        -------
        Dict[str, Any]
            Changes. If no changes, dict will be empty
        """
        diff = changes(self, other)
        if diff == NOCHANGE:
            return dict()
        return diff
    
    def prepare(self) -> None:
        """Performs preparations on the config. Gets typically invoked by the training runners.
        """
        if self.diff_config is not None:
            if isinstance(self.diff_config, str) and os.path.exists(self.diff_config):
                try:
                    self.diff_config = JsonConvertible.load_from_file(self.diff_config)
                except Exception as err:
                    logging.exception(f"Could not load diff config from path {self.diff_config} for config name: {self.name_experiment}")
