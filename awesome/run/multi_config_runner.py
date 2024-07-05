from dataclasses import field
from datetime import datetime
from typing import List, Optional, Tuple, Type
from awesome.run.multi_runner import MultiRunner
from awesome.run.multi_config_config import MultiConfigConfig
import itertools
import logging
import copy
from awesome.error import ArgumentNoneError
from awesome.serialization.json_convertible import JsonConvertible
from awesome.util.diff import changes, NOCHANGE
import os
import re

from awesome.util.path_tools import relpath


class MultiConfigRunner(MultiRunner):
    """Creates multiple runner based on multiple given configs."""
    
    config: MultiConfigConfig

    def __init__(self,
                 config: MultiConfigConfig,
                 **kwargs) -> None:
        super().__init__(config=config,
                         **kwargs)

    def scan_dir(self, directory, pattern, recursive: bool = False, depth: int = 100) -> List[str]:
        res = []
        if not os.path.exists(directory):
                raise FileNotFoundError(f"Config directory {directory} does not exist.")
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            if os.path.isfile(path):
                match = pattern.fullmatch(file)
                if match is not None:
                    res.append(path)
            elif os.path.isdir(path):
                if recursive and depth >= 0:
                    results = self.scan_dir(path, pattern, recursive=recursive, depth=depth-1)
                    res.extend(results)
        return res

    def get_config_paths(self) -> List[str]:
        if self.config.mode == 'plain':
            return self.config.config_paths
        elif self.config.mode == 'scan_dir' or self.config.mode == 'scan_dir_recursive':
            ret = []
            directory = self.config.scan_config_directory
            pattern = re.compile(self.config.config_pattern)
            ret = self.scan_dir(directory, pattern, recursive=(self.config.mode == "scan_dir_recursive"))
            return ret
        else:
            raise ValueError(f"mode must be either 'plain' or 'scan_dir' but is {self.config.mode}")
        
    def save_child_configs(self, directory: str) -> List[str]:
        """Saves the configs of the child runners to a directory.

        Parameters
        ----------
        directory : str
            The directory where the configs will be saved.

        Returns
        -------
        List[str]
            The paths where the configs are stored.
        """
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save configs
        num_digits = len(str(len(self.child_configs)))
        num_fmt = f"{{:0{num_digits}d}}"
        paths = []
        for i, config in enumerate(self.child_configs):
            fmt = f"#{num_fmt}_{config.name_experiment}.yaml"
            path = os.path.join(directory, fmt.format(i))
            path = config.save_to_file(path, no_uuid=True, no_large_data=True, override=True)
            paths.append(path)
        return paths

    def create_jobs(self, ref_dir: Optional[str] = None, preset_output_folder: bool = False) -> List[Tuple[str, List[str]]]:
        created_at = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        is_from_file = ref_dir is not None
        if ref_dir is None:
            ref_dir = os.getcwd()
        ref_dir = os.path.abspath(ref_dir)
        if ref_dir != self.__jobsrefdir__:
            self.__jobsrefdir__ = ref_dir
            self.__jobs__ = None
        if self.__jobs__ is not None:
            return self.__jobs__
        if self.config.config_directory is None:
            raise ArgumentNoneError("config_directory")
        config_directory = os.path.abspath(self.config.config_directory)
        runner_script_path = os.path.abspath(self.config.runner_script_path)
        if not os.path.exists(runner_script_path):
            raise FileNotFoundError(
                f"Runner script not found at {runner_script_path}")

        child_config_paths = self.save_child_configs(config_directory)
        runner_script_path = os.path.abspath(runner_script_path)

        rel_child_config_paths = [relpath(self.__jobsrefdir__, p,
                             is_from_file=is_from_file) for p in child_config_paths]

        num_digits = len(str(len(self.child_configs)))
        num_fmt = f"{{:0{num_digits}d}}"

        items = [
            (relpath(self.__jobsrefdir__, runner_script_path, is_from_file=is_from_file),
             [f"--config-path", config_path, "--name-experiment", f"{num_fmt.format(i)}_{_name}"]) for i,
            (_name, config_path) in enumerate(zip([x.name_experiment for x in self.child_configs], rel_child_config_paths))]
        
        items = []
        for i, (_name, config_path) in enumerate(zip([x.name_experiment for x in self.child_configs], rel_child_config_paths)):
            name_experiment = f"#{num_fmt.format(i)}_{_name}"
            output_folder = None
            if preset_output_folder:
                if self.child_configs[i].output_folder is not None:
                    output_folder = self.child_configs[i].output_folder
                else:
                    path = os.path.join(self.child_configs[i].runs_path, name_experiment + "_" + created_at)
                    output_folder = path
            
            item = self._generate_single_job(
               runner_script_path=runner_script_path,
               is_ref_dir_from_file=is_from_file,
               experiment_name=_name,
                config_path=config_path,
                output_folder=output_folder
               )
            items.append(item)

        self.__jobs__ = items
        return items

    def build(self, build_children: bool = True, **kwargs) -> None:
        configs = self.get_config_paths()

        for config_path in configs:
            config = JsonConvertible.load_from_file(config_path)
            rnr = self.runner_type(config=config)
            rnr.parent = self
            self.child_runners.append(rnr)

        # Build children
        if build_children:
            for runner in self.child_runners:
                runner.build()
