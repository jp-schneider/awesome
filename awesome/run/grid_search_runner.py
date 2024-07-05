from dataclasses import field
from typing import Type
from awesome.run.multi_runner import MultiRunner
from awesome.run.config import Config, GridSearchConfig
import itertools
import logging
import copy
from awesome.error import ArgumentNoneError
from awesome.util.diff import changes, NOCHANGE


class GridSearchRunner(MultiRunner):
    """Creates multiple child runners by doing a cartesian product of the param_grid of the config."""

    def __init__(self,
                 config: GridSearchConfig,
                 **kwargs) -> None:
        super().__init__(config=config,
                         **kwargs)

    def build(self, build_children: bool = True, **kwargs) -> None:
        # Build the config for each child runner by doing a cartesian product of the param_grid and insert it in base config
        keys = self.config.param_grid.keys()
        values = self.config.param_grid.values()
        for v in itertools.product(*values):
            
            base_configs = self.base_config

            if not isinstance(self.base_config, list):
                base_configs = [self.base_config]
            for base_config in base_configs:
                # Copy base config
                config = copy.deepcopy(base_config)
                # Insert values
                for k, v_ in zip(keys, v):
                    setattr(config, k, v_)
                rnr = self.runner_type(config=config)
                # Create magic property diff-config to directly indicate the difference between the base config and the child config
                rnr.diff_config = dict()
                for k, v_ in zip(keys, v):
                    chg = changes(getattr(base_config, k), v_)
                    if chg != NOCHANGE:
                        rnr.diff_config[k] = chg
                rnr.parent = self
                rnr.config.diff_config = copy.deepcopy(rnr.diff_config)
                self.child_runners.append(rnr)
        # Build children
        if build_children:
            for runner in self.child_runners:
                runner.build()
