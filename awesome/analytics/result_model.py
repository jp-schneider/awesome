import copy
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Literal, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy import iterable
import torch
from awesome.serialization import JsonConvertible
import pandas as pd
import re
from PIL import Image
from awesome.agent.agent import Agent, Tracker
from awesome.agent.torch_agent_checkpoint import TorchAgentCheckpoint

from awesome.util.reflection import dynamic_import
from awesome.util.path_tools import open_folder
from awesome.run.config import Config
from awesome.util.diff import flatten
from awesome.run.runner import Runner
from awesome.run.awesome_runner import AwesomeRunner
from awesome.util.alter_working_directory import AlterWorkingDirectory
from awesome.util.temporary_property import TemporaryProperty
import awesome.run.functions as F

def minmax(v: torch.Tensor, new_min: float = 0., new_max: float = 1.):
    v_min, v_max = v.min(), v.max()
    return (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


def mean_std_norm(v: torch.Tensor, mu: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
    if mu is None:
        mu = v.mean()
    if std is None:
        std = v.std()
    return (v - mu) / std


def inverse_mean_std_norm(v: torch.Tensor, mu: float, std: float) -> torch.Tensor:
    return (v * std) + mu


def get_circular_index(vals: iterable, index: int) -> int:
    if index < 0:
        index = (-1) * (abs(index) % len(vals))
        if index < 0:
            index = len(vals) + index
    return index
    
@dataclass
class ResultModelConfig(JsonConvertible):

    name: str = field(default=None)

    number: int = field(default=None)
    """A number which is assigned for identification."""

    result_directory: str = field(default="final_mask")

    additional_result_directories: List[str] = field(default_factory=lambda : ["prior_mask"])

    epochs: List[int] = field(default_factory=list)
    """Epochs used within the model."""

    image_indices: List[int] = field(default_factory=list)
    """Image indices used within the model."""

@dataclass
class ResultModel():

    config: ResultModelConfig

    output_directory: str

    numbering: bool = True

    MASK_IMAGE_PATTERN = r"mask_(?P<image_index>\d+)_ep_(?P<epoch>[0-9]+)\.png"
    MASK_IMAGE_FORMAT = "mask_{}_ep_{}.png"

    RAW_MASK_IMAGE_PATTERN = r"unary_(?P<image_index>\d+)_ep_(?P<epoch>[0-9]+)\.tif"
    RAW_MASK_IMAGE_FORMAT = "unary_{}_ep_{}.tif"

    PRIOR_MASK_IMAGE_PATTERN = r"prior_mask_(?P<image_index>\d+)_ep_(?P<epoch>[0-9]+)\.png"
    PRIOR_MASK_IMAGE_FORMAT = "prior_mask_{}_ep_{}.png"

    RAW_PRIOR_MASK_IMAGE_PATTERN = r"prior_unary_(?P<image_index>\d+)_ep_(?P<epoch>[0-9]+)\.tif"
    RAW_PRIOR_MASK_IMAGE_FORMAT = "prior_unary_{}_ep_{}.tif"

    CHECKPOINT_PATTERN = r"checkpoint_epoch_(?P<epoch>[0-9]+)\.pth"
    CHECKPOINT_FORMAT = "checkpoint_epoch_{}.pth"

    PRIOR_CHECKPOINT_PATTERN = r"prior_cache_epoch_(?P<epoch>[0-9]+)\.pth"
    PRIOR_CHECKPOINT_FORMAT = "prior_cache_epoch_{}.pth"

    index: pd.DataFrame = field(default_factory=lambda: ResultModel.create_index_df())

    checkpoint_index: pd.DataFrame = field(default_factory=lambda: ResultModel.create_checkpoint_df())

    _run_config: Optional[Config] = field(default=None)
    """If experiment was done using a runner, there exists a run config in the folder which can be loaded."""

    _runners: Dict[int, Runner] = field(default_factory=dict)

    run_config_path: Optional[str] = field(default=None)
    """Path to the run config."""

    RUN_CONFIG_PATTERN = "init_cfg_(?P<config_name>[a-zA-Z0-9_]+)\.yaml"

    rerun_parent: 'ResultModel' = field(default=None)
    """If the current ResultModel was created by a rerun, this is the parent ResultModel."""

    getitem_mask_mode: Literal['mask', 'prior_mask', 'both'] = field(default='mask')

    getitem_use_raw: bool = field(default=False)

    getitem_epoch: int = -1
    
    features: List[str] = field(default_factory=list)
    """Highlight features for the result model."""

    @classmethod
    def create_index_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        return pd.DataFrame(data, columns=["epoch", "image_index", "path", "path_raw", "info", "prior_path", "prior_path_raw"])

    @classmethod
    def create_checkpoint_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=["epoch", "path", "checkpoint", "agent", "prior_cache_path"])
        df['checkpoint'] = df['checkpoint'].astype(object)
        df['agent'] = df['agent'].astype(object)
        return df

    def __repr__(self) -> str:
        self_dict = dict(vars(self))
        dont_show = ["index", "checkpoint_index"]
        for prop in dont_show:
            if prop in self_dict and self_dict.get(prop) is not None:
                self_dict[prop] = "[...]"
        return type(self).__name__ + f"({', '.join([k+'='+str(v) for k, v in self_dict.items()])})"

    def open_folder(self) -> None:
        """Opens the model output path in the operating system file explorer.

        Parameters
        ----------
        result_type : Optional[Union[ResultType, str]], optional
            The result type path to open. None opens base directory, dedicated result_types will open their own folder., by default None
        """
        path = os.path.normpath(self.output_path)
        open_folder(path)


    def index_from_epoch(self, epoch: int):
        """Returns the index for an epoch to use within getitem."""
        if epoch not in self.config.epochs:
            raise ValueError(f"Invalid epoch: {epoch}. Avail: {', '.join([str(i) for i in self.config.epochs])}")
        return self.config.epochs.index(epoch)

    def compute_diff_string(self, 
                            default_values: Dict[str, Any] = None, 
                            key_alias: Optional[Dict[str, str]] = None,
                            value_alias: Optional[Dict[str, str]] = None,
                            value_separator: str = ":",
                            item_separator: str = "_",
                            use_diff_config: bool = True,
                            nested_keys: Optional[Dict[str, Any]] = None,
                            specified_markers: bool = True
                            ) -> str:
        """Computes a diff string based on the current run config and the default values.
        This string will show the declaration differences of the current model w.r.t the default values.

        Parameters
        ----------
        default_values : Dict[str, Any], optional
            Default values to compare against, by default None
        key_alias : Optional[Dict[str, str]], optional
            Alias name for the key. This should be the mapping of the old key name to the new one.
            As the default values can be a nested dict, this nesting can be adressed by concatenating the keys with the separator '__'.
            So the key_alias itself is a flat dict, by default None
        value_alias : Optional[Dict[str, str]], optional
            This can be used to alter the value of the differing key by simple substitution.
            Like 'False': 'No', by default None  
            , by default None
        value_separator : str, optional
            How the value should be seperated from the key in the result string, by default ":"
        item_separator : str, optional
            How individual items should be separated, by default "_"
        use_diff_config : bool, optional
            If the diff config should be used or the whole run config, by default True
        nested_keys : Optional[Dict[str, Any]], optional
            If only a subset of the keys should be used for the diff, by default None
            Only valid when use_diff_config is False.
        specified_markers : bool, optional
            If the diff string should contain markers for specified values and changed values, by default True
        Returns
        -------
        str
            The diffstring with all altered parameters and their values.
        """        
        res = dict()

        def _process_value(v: Any, default_value: Optional[Any], 
                           value_alias: Optional[Dict[str, str]] = None
                           ) -> Any:
            if isinstance(v, dict) or isinstance(default_value, dict):
                ret = dict()
                keys = []
                if v is not None:
                    keys = list(v.keys())
                if default_value is not None:
                    keys += list(default_value.keys())
                keys = set(keys)
                for k in keys:
                    _v = v.get(k, None) if v is not None else None
                    _def = default_value.get(k, None) if default_value is not None else None
                    name = k
                    if _v is None and _def is None:
                        continue
                    ret[name] = _process_value(_v, _def, value_alias=value_alias)
                return ret
            else:
                has_changes = None
                if v is not None:
                    if default_value is not None:
                        has_changes = not (v == default_value)
                    v_str = str(v)
                    if value_alias is not None:
                        v_str = value_alias.get(v_str, v_str)
                    if has_changes is not None:
                        v_str = (v_str + ("*" if specified_markers else "")) if has_changes else v_str
                    else:
                        v_str = v_str + ("?" if specified_markers else "")
                    return v_str
                elif default_value is not None:
                    v_str = str(default_value) 
                    if value_alias is not None:
                        v_str = value_alias.get(v_str, v_str)
                    if specified_markers:
                        v_str += "-"
                    return v_str
                else:
                    raise NotImplementedError("Either v, or default should not be None, there is an error.")

        own = None
        if use_diff_config:
            own = self.run_config.diff_config
        else:
            from awesome.util.diff import changes, filter, nested_keys as get_nested_keys
            own = self.run_config.to_json_dict(no_uuid=True, no_large_data=True)
            if nested_keys is not None:
                own = filter(own, nested_keys)
                default_values = filter(default_values, nested_keys)
            else:
                altered = changes(default_values, own)
                own = filter(own, get_nested_keys(altered))
                default_values = filter(default_values, get_nested_keys(altered))
            

        deep = _process_value(own, default_values, value_alias)
        flattened_dict = flatten(deep, separator="__")
        if key_alias is not None:
            ret = {}
            for k, v in flattened_dict.items():
                new_key = key_alias.get(k, k)
                ret[new_key] = v
            flattened_dict = ret
        comp = [k + value_separator + v for k, v in flattened_dict.items()]
        return item_separator.join(comp)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> torch.Tensor:
        index = get_circular_index(self.config.image_indices, index)
        index = self.config.image_indices[index]

        epoch = get_circular_index(self.config.epochs, self.getitem_epoch)
        epoch = self.config.epochs[epoch]

        data = self.index
        row = data[(data['epoch'] == epoch) & (data['image_index'] == index)].iloc[0]

        mask = None
        prior_mask = None

        if self.getitem_mask_mode == 'mask' or self.getitem_mask_mode == 'both':
            if self.getitem_use_raw:
                mask = F.load_mask_single_channel(row['path_raw'])
            else:
                mask, _ = F.load_mask_multi_channel(row['path'])
        if self.getitem_mask_mode == 'prior_mask' or self.getitem_mask_mode == 'both':
            if self.getitem_use_raw:
                prior_mask = F.load_mask_single_channel(row['prior_path_raw'])
            else:
                prior_mask, _ = F.load_mask_multi_channel(row['prior_path'])
        if self.getitem_mask_mode == 'mask':
            return mask
        elif self.getitem_mask_mode == 'prior_mask':
            return prior_mask
        else:
            return mask, prior_mask

    @F.saveable()
    def plot_mask(self, 
                  index: int, 
                  epoch: int = -1,
                  mode: Literal['mask', 'prior_mask'] = 'mask',
                  tight: bool = False,
                  ) -> AxesImage:
        data = None
        runner = self.get_runner(epoch)
        image = runner.dataloader.get_image(index)
        with TemporaryProperty(self, getitem_mask_mode=mode, getitem_epoch=epoch):
            data = self[index]
        
        return F.plot_mask(image, data, tight=tight)

        

    
    def get_tracked_metrics(self, index: int = -1) -> List[str]:
        """Get a list of tracked metric names in the tracker of the 'index' checkpoint.

        Parameters
        ----------
        index : int, optional
            Index of the checkpoint to get the metric from. Default to -1, meaning last.

        Returns
        -------
        List[str]
            List of metric tags which are tracked.
        """
        tracker = self.get_tracker(-1)
        return list(tracker.metrics.keys())

    @staticmethod
    def get_metric_display_name(metric: str) -> str:
        if metric == "total_variation":
            return "TV"
        return metric.upper()

    @property
    def display_name(self) -> str:
        """Get a name for the result."""
        name = self.config.name or os.path.basename(self.output_directory)
        if self.numbering and self.config.number is not None:
            name = str(self.config.number) + ". " + name
        return name

    @display_name.setter
    def display_name(self, value: str):
        self.config.name = value

    @property
    def name(self) -> str:
        return os.path.basename(self.output_directory)

    @property
    def number(self) -> int:
        if self.config is None:
            return 0
        return self.config.number

    @property
    def run_config(self) -> Optional[Config]:
        """Gets the config which was used to run the experiment, when it was executed with a runner.

        Returns
        -------
        Optional[Config]
            Config of the experiment.
        """
        if self._run_config is None:
            if self.run_config_path is not None:
                try:
                    self._run_config = JsonConvertible.load_from_file(self.run_config_path)
                except Exception as err:
                    logging.exception(f"Could not load run config from {self.run_config_path}.")
        return self._run_config

    @run_config.setter
    def run_config(self, value: Config):
        self._run_config = value

    @classmethod
    def from_path(cls, path: str) -> 'ResultModel':
        if not os.path.exists(path):
            raise ValueError(f"Path: {path} does not exists. Can not create Resultmodel.")
        config = None
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            config = ResultModelConfig.load_from_file(config_path)
        else:
            config = ResultModelConfig(name=os.path.basename(path))
        model = cls(config=config, output_directory=path)
        model.create_index()
        model.save_config()
        return model

    def reload_config(self):
        """Reloads the config from the config.json file. Will override the existing one."""
        config_path = os.path.join(self.output_directory, "config.json")
        self.config = ResultModelConfig.load_from_file(config_path)

    def save_config(self):
        self.config.save_to_file(self.config_path, override=True)


    def scan_result_directory(self, path: str) -> List[Dict[str, Any]]:
        data = []
        mask_data = self.read_directory(path, ResultModel.MASK_IMAGE_PATTERN)
        prior_mask_data = self.read_directory(path, ResultModel.PRIOR_MASK_IMAGE_PATTERN)

        raw_mask_data = []
        raw_prior_mask_data = []

        raw_path = os.path.join(path, "raw")
        if os.path.exists(raw_path):
            raw_mask_data = self.read_directory(raw_path, ResultModel.RAW_MASK_IMAGE_PATTERN)
            raw_prior_mask_data = self.read_directory(raw_path, ResultModel.RAW_PRIOR_MASK_IMAGE_PATTERN)

        mask_dict = {str(v['image_index']) + ":" + str(v['epoch']): v for v in mask_data}
        prior_mask_dict = {str(v['image_index']) + ":" + str(v['epoch']): v for v in prior_mask_data}

        raw_mask_dict = {str(v['image_index']) + ":" + str(v['epoch']): v for v in raw_mask_data}
        raw_prior_mask_dict = {str(v['image_index']) + ":" + str(v['epoch']): v for v in raw_prior_mask_data}

        # Combine mask and prior mask based on index and epoch
        for key, val in mask_dict.items():
            v = dict(val)
            
            prior_mask = prior_mask_dict.get(key, None)
            if prior_mask is not None:
                v['prior_path'] = prior_mask['path']
            
            raw_mask = raw_mask_dict.get(key, None)
            if raw_mask is not None:
                v['path_raw'] = raw_mask['path']

            raw_prior_mask = raw_prior_mask_dict.get(key, None)
            if raw_prior_mask is not None:
                v['prior_path_raw'] = raw_prior_mask['path']
            
            data.append(v)
        return data


    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        ckps = self.read_directory(self.output_directory, ResultModel.CHECKPOINT_PATTERN, False, keys=["epoch"])
        priors = {int(x['epoch']): x for x in self.read_directory(self.output_directory, ResultModel.PRIOR_CHECKPOINT_PATTERN, False, keys=["epoch"])}

        for ckp in ckps:
            prior = priors.get(ckp['epoch'], None)
            if prior is not None:
                ckp['prior_cache_path'] = prior['path']
            else:
                ckp['prior_cache_path'] = None
                logging.warning(f"Prior cache not found for checkpoint: {ckp['epoch']} model {self.name}!")
        return ckps

    def create_index(self):
        data = self.scan_result_directory(self.result_path)
        for path in self.additional_result_paths:
            data += self.scan_result_directory(path)

        # Sort by epoch and image index
        data = sorted(data, key=lambda x: (x['epoch'], x['image_index']))

        self.index = type(self).create_index_df(data)
        if 'epoch' in self.index.columns:
            self.config.epochs = [int(x) for x in sorted(self.index['epoch'].unique())]
        else:
            self.config.epochs = []

        if 'image_index' in self.index.columns:
            self.config.image_indices = [int(x) for x in sorted(self.index['image_index'].unique())]
        else:
            self.config.image_indices = []

        # Checking for init config
        init_cfg = self.read_directory(self.output_directory, ResultModel.RUN_CONFIG_PATTERN, False)
        if len(init_cfg) > 0:
            self.run_config_path = init_cfg[0]['path']
        else:
            self.run_config_path = None
            self.run_config = None

        self.checkpoint_index = type(self).create_checkpoint_df(self.scan_checkpoints()).sort_values("epoch")


    def load_image(self, path: str) -> torch.Tensor:
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        with Image.open(path) as img:
            tens = to_tensor(img)
            return tens


    def read_directory(self, 
                       path: str, 
                       pattern: str, 
                       has_info: bool = True, 
                       keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if keys is None:
            keys = ["epoch", "image_index"]
        key_data = {key: None for key in keys}
        res = []
        for file in os.listdir(path):
            match = re.fullmatch(pattern=pattern, string=file)
            if match:
                data = copy.deepcopy(key_data)
                for k in data.keys():
                    try:
                        v = int(match.group(k))
                        data[k] = v
                    except IndexError as err:
                        pass
                entry = dict(path=os.path.join(path, file), **data)
                if has_info:
                    base, ext = os.path.splitext(file)
                    result_info = os.path.join(path, base + ".json")
                    if not os.path.exists(result_info):
                        result_info = None
                    entry['info'] = result_info
                res.append(entry)
        return res

    def get_checkpoint(self, index: int) -> TorchAgentCheckpoint:
        """Gets the agent checkpoint from the given index.

        Parameters
        ----------
        index : int
            The index of checkpoints.

        Returns
        -------
        TorchAgentCheckpoint
            Checkpoint
        """
        if len(self.checkpoint_index) == 0:
            raise IndexError("No checkpoints found!")
        index = get_circular_index(self.checkpoint_index.index, index)
        ckp_row = self.checkpoint_index.iloc[index]
        col_iloc = self.checkpoint_index.columns.get_loc("checkpoint")
        if pd.isna(ckp_row['checkpoint']):
            checkpoint = TorchAgentCheckpoint.load(ckp_row['path'])
            self.checkpoint_index.iloc[index, col_iloc] = checkpoint
        return self.checkpoint_index.iloc[index, col_iloc]

    def get_state_dict(self, index: int) -> Optional[Dict[str, Any]]:
        """Gets the state dict from the given index.

        Parameters
        ----------
        index : int
            The index of checkpoints.

        Returns
        -------
        Dict[str, Any]
            The state dict.
        """
        ckp = self.get_checkpoint(index)
        if ckp is None:
            return None
        return ckp.model_state_dict

    def get_agent(self, index: int) -> Agent:
        """Gets the agent from the given index.

        Parameters
        ----------
        index : int
            The index of checkpoints.

        Returns
        -------
        Agent
            The restored agent.
        """
        index = get_circular_index(self.checkpoint_index.index, index)
        ckp_row = self.checkpoint_index.iloc[index]
        col_iloc = self.checkpoint_index.columns.get_loc("agent")
        if pd.isna(ckp_row['agent']):
            checkpoint = self.get_checkpoint(index)
            agent = checkpoint.to_agent()
            self.checkpoint_index.iloc[index, col_iloc] = agent
        return self.checkpoint_index.iloc[index, col_iloc]

    def get_prior_cache_path(self, index: int) -> Optional[str]:
        """Gets the prior cache path from the given index if present.

        Parameters
        ----------
        index : int
            The index of checkpoints.

        Returns
        -------
        str
            The path to the prior chache.
        """
        index = get_circular_index(self.checkpoint_index.index, index)
        ckp_row = self.checkpoint_index.iloc[index]
        val = ckp_row.get('prior_cache_path', None)
        if val == float('nan'):
            val = None
        return val

    def get_runner(self, index: int = -1) -> Runner:
        index = get_circular_index(self.checkpoint_index.index, index)
        if self._runners is None:
            self._runners = dict()
        if index not in self._runners:
            cfg = self.run_config
            cfg.output_folder = self.output_directory
            runner_type = dynamic_import(cfg.used_runner_type) if cfg.used_runner_type is not None else AwesomeRunner
            
            with AlterWorkingDirectory(os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))):
                self._runners[index] = runner_type(config=cfg)
                self._runners[index].build()
                try:
                    # Load prior cache if exists
                    prior_cache_path = self.get_prior_cache_path(index)
                    if prior_cache_path is not None:
                        self._runners[index].dataloader.prior_load(prior_cache_path)
                    else:
                        if self._runners[index].config.use_prior_model:
                            logging.warning(f"Prior cache not found for index: {index}!")

                    agent = self.get_agent(index=index)
                    self._runners[index].patch_agent(agent)
                except Exception as err:
                    logging.exception(f"Could not patch agent to runner.")
        return self._runners.get(index, None)


    @property
    def output_path(self) -> str:
        path = os.path.join(self.output_directory)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def result_path(self) -> str:
        path = os.path.join(self.output_directory, self.config.result_directory)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def additional_result_paths(self) -> List[str]:
        paths = []
        for p in self.config.additional_result_directories:
            path = os.path.join(self.output_directory, p)
            if os.path.exists(path):
                paths.append(path)
        return paths

    @property
    def config_path(self) -> str:
        path = os.path.join(self.output_directory, "config.json")
        return path


    def get_tracker(self, index: int) -> Tracker:
        """Gets the tracker of the given checkpoint index.

        Parameters
        ----------
        index : int
            the checkpoint index where the tracker should be loaded from.

        Returns
        -------
        Tracker
            The tracker containing state and metric info.
        """
        return self.get_checkpoint(index).tracker
