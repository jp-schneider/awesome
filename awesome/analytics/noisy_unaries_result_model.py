
from awesome.analytics.result_model import ResultModel
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

class NoisyUnariesResultModel(ResultModel):

    @classmethod
    def create_checkpoint_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=["epoch", "path", "checkpoint", "agent", "prior_cache_path", "noisy_unaries_path"])
        df['checkpoint'] = df['checkpoint'].astype(object)
        df['agent'] = df['agent'].astype(object)
        return df
    
    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        ckps = super().scan_checkpoints()
        noisy_unaries_dict_path = [x for x in self.read_directory(self.output_directory, r"noisy_unaries_dict.pth?", False, keys=None)]
        if len(noisy_unaries_dict_path) == 0:
            logging.warning(f"No noisy_unaries_dict.pth found in {self.output_directory}")
        else:
            noisy_unaries_dict_path = noisy_unaries_dict_path[0]
            for ckp in ckps:
                ckp['noisy_unaries_path'] = noisy_unaries_dict_path.get("path")
        return ckps
    
    def get_noisy_unaries_dict(self, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
        ckp = self.checkpoint_index.iloc[0]
        return torch.load(ckp['noisy_unaries_path'], map_location=map_location)