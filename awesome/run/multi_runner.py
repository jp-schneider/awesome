from dataclasses import field
import logging
from typing import List, Type, Optional, Tuple

from matplotlib import pyplot as plt
from awesome.run.config import Config
from awesome.run.runner import Runner
import os
from awesome.error import ArgumentNoneError
from awesome.util.reflection import dynamic_import
from awesome.run.multi_runner_config import MultiRunnerConfig
from datetime import datetime
from awesome.serialization.json_convertible import JsonConvertible
from awesome.util.path_tools import relpath

class MultiRunner(Runner):
    """A runner which can run multiple child runners. Typically used to find the best hyperparameters."""

    child_runners: List[Runner]
    """Child runners which will be run / trained."""

    runner_type: Type
    """The type of the runner which will be used for the child runners."""

    base_config: Config
    """The base config which will be used to create the child configs."""

    __jobs__: List[Tuple[str, List[str]]]
    """List of jobs which will be executed."""

    __jobsrefdir__: str
    """Reference directory for the jobs."""

    __date_created__: str
    """Date when the runner was created."""

    def __init__(self,
                 config: MultiRunnerConfig,
                 **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if config.runner_type is None:
            raise ArgumentNoneError("runner_type")
        rt = dynamic_import(config.runner_type)
        if not issubclass(rt, Runner):
            raise TypeError("runner_type must be a subclass of Runner")
        self.runner_type = rt
        base_config = self.config.base_config
        if base_config is not None:
            if isinstance(base_config, str):
                base_config = JsonConvertible.load_from_file(base_config)
            if not isinstance(base_config, Config):
                raise TypeError("base_config must be a subclass of Config")
        else:
            base_config = self.config
        self.base_config = base_config
        self.child_runners = []
        self.__jobs__ = []
        self.__date_created__ = None
        self.__jobsrefdir__ = None

    @property
    def date_created(self) -> str:
        """Returns the date when the runner was created."""
        if self.__date_created__ is None:
            self.__date_created__ = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        return self.__date_created__

    @property
    def child_configs(self) -> List[Config]:
        """Returns the configs of the child runners."""
        return [runner.config for runner in self.child_runners]

    def build(self, build_children: bool = True, **kwargs) -> None:
        pass

    def save_child_configs(self, directory: str, prefix: Optional[str] = None) -> List[str]:
        """Saves the configs of the child runners to a directory.

        Parameters
        ----------
        directory : str
            The directory where the configs will be saved.

        prefix : Optional[str], optional
            The prefix which will be added to the config names, by default None

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
        fmt = (f"{prefix}_" if prefix is not None else "") + f"config_{num_fmt}.yaml"
        paths = []
        for i, config in enumerate(self.child_configs):
            path = os.path.join(directory, fmt.format(i))
            path = config.save_to_file(path, no_uuid=True, no_large_data=True)
            paths.append(path)
        return paths

    def create_job_file(self) -> str:
        """Creates a job file for slurm cluster.

        Returns
        -------
        str
            Path to the job file.
        """
        if self.config.config_directory is None:
            raise ArgumentNoneError("config_directory")
        config_directory = os.path.abspath(self.config.config_directory)
        preset_output_folder = self.config.preset_output_folder
        job_file_path = self.config.job_file_path

        exp_name_date = f"{self.base_config.name_experiment}_{self.date_created}"

        # Create job file
        if job_file_path is None or (
                len(os.path.basename(job_file_path)) == 0) or (
                '.py' not in os.path.basename(job_file_path)):
            if job_file_path is None:
                job_file_path = config_directory
            job_file_path = os.path.join(
                job_file_path,
                f"JobFile_{exp_name_date}.py")
        if not os.path.exists(os.path.dirname(job_file_path)):
            os.makedirs(os.path.dirname(job_file_path))

        items = [str(x) for x in self.create_jobs(preset_output_folder=preset_output_folder)]
        formatted_items = (', ' + os.linesep + '\t').join(items)
        content = (f"from typing import List, Tuple" + os.linesep +
                   "JOBS: List[Tuple[str, List[str]]] = [" +
                   formatted_items + os.linesep +
                   "]")
        with open(job_file_path, "w") as f:
            f.write(content)
        return job_file_path

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

        exp_name_date = f"{self.base_config.name_experiment}_{self.date_created}"

        paths = self.save_child_configs(config_directory, exp_name_date)
        runner_script_path = os.path.abspath(runner_script_path)

        rel_paths = [relpath(self.__jobsrefdir__, p,
                             is_from_file=is_from_file) for p in paths]

        items = []
        for i, p in enumerate(rel_paths):
            # exec_file = relpath(self.__jobsrefdir__, runner_script_path, is_from_file=is_from_file)
            # args = [
            #     f"--config-path", p, 
            #     "--name-experiment", f"{self.base_config.name_experiment}_#{i}"
            #   ]
            # if preset_output_folder:
            #     output_folder = self.base_config.output_folder
            #     path = os.path.join(self.base_config.output_folder, f"#{i}")
            #     args += [
            #         "--output-folder", f"{self.base_config.output_folder}_#{i}"
            #     ]
            # items.append((exec_file, args))
            output_folder = None
            experiment_name = f"{self.base_config.name_experiment}_{i}"

            if preset_output_folder:
                if self.base_config.output_folder is not None:
                    output_folder = self.base_config.output_folder
                else:
                    path = os.path.join(self.base_config.runs_path, experiment_name + "_" + created_at)
                    output_folder = path

            item = self._generate_single_job(
                runner_script_path=runner_script_path,
                is_ref_dir_from_file=is_from_file,
                experiment_name=experiment_name,
                config_path=p,
                output_folder=output_folder
            )
            items.append(item)

        self.__jobs__ = items
        return items
    
    def _generate_single_job(self, 
            runner_script_path: str,
            is_ref_dir_from_file: bool,
            experiment_name: str,
            config_path: str,
            output_folder: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        exec_file = relpath(self.__jobsrefdir__, runner_script_path, is_from_file=is_ref_dir_from_file)
        args = [
            f"--config-path", config_path, 
            "--name-experiment", experiment_name
            ]
        if output_folder is not None:
            args += [
                "--output-folder", os.path.normpath(output_folder)
            ]
        return (exec_file, args)

    def child_runner_commands(self) -> List[Tuple[Runner, str]]:
        """Returns a list of tuples which contain the child runner and the command for the child runner in a seperate process.

        Returns
        -------
        List[Tuple[Runner, str]]
            List of subrunners and command to execute.
        """
        jobs = self.create_jobs()
        return list(zip(self.child_runners, [f'python {x} {" ".join(y)}' for x, y in jobs]))

    def train(self, *args, **kwargs):
        runner = self
        config = self.config
        for i, child_runner in enumerate(runner.child_runners):
            try:
                cfg = child_runner.config
                logging.info(f"Building child runner #{i}...")
                child_runner.build()
                # Save config and log it
                cfg_file = child_runner.store_config()
                child_runner.log_config()
                logging.info(f"Stored config in: {cfg_file}")
                logging.info(
                    f"Training with child runner #{i} {cfg.name_experiment} with diff-config: \n{JsonConvertible.convert_to_yaml_str(child_runner.diff_config, no_large_data=True, no_uuid=True)}")
                with plt.ioff():
                    child_runner.train()
                logging.info(f"Training done with child runner #{i} {cfg.name_experiment}")  
            except Exception as err:
                logging.exception(f"Raised {type(err).__name__} in training child runner #{i}")