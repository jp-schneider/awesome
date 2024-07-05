
from typing import Any, Dict
from awesome.mixin.argparser_mixin import ArgparserMixin
from dataclasses import dataclass, field

from awesome.serialization.json_convertible import JsonConvertible


@dataclass
class SetupProgress(JsonConvertible):
    pass


@dataclass
class SetupConfig(JsonConvertible, ArgparserMixin):
    """This is the config class for."""

    data_path: str = field(default="./data")
    """Path to the data folder."""

    checkpoint_path: str = field(default="./data/checkpoints")
    """Path to the checkpoint folder."""

    dataset_path: str = field(default="./data/datasets")
    """Path to the dataset folder."""

    third_party_code_path: str = field(default="./third_party")
    """Path to the third party code folder."""

    feature_input_setup: bool = field(default=True)
    """If the soft segmentation models and checkpoints should be downloaded and set up."""

    download_convexity_data: bool = field(default=True)
    """If the convexity data should be downloaded and set up (needed for convexity experiments)."""

    download_fbms_data: bool = field(default=True)
    """If the fbms data should be downloaded and set up (needed for path-connected experiments)."""

    download_uncertainty_multicut_models: bool = field(default=True)
    """Downloads pre-trained models from "Uncertainty in Minimum Cost Multicuts for Image and Motion Segmentation" (https://arxiv.org/abs/2105.07469)."""

    download_uncertainty_multicut_labels: bool = field(default=True)
    """Downloads the labels for the uncertainty in minimum cost multicuts models."""

    download_soft_semantic_segmentation_models: bool = field(default=True)
    """Downloads the soft segmentation code and models (https://github.com/iyah4888/SIGGRAPH18SSS.git)."""

    download_pc_pretrain_states: bool = field(default=True)
    """If pre-trained states should be downloaded. These are excpecially used for path-connectedness experiments."""

    delete_zip_files: bool = field(default=True)
    """If the zip files should be deleted after extraction."""

    def get_urls(self) -> Dict[str, Any]:
        return {
            "convexity_dataset": "https://vision.cs.uwaterloo.ca/files/ConvexityDB.zip",
            "convexity_dataset_feat": "https://uni-siegen.sciebo.de/s/NPOkXURyN3SDzHo/download",
            "fbms_train_dataset": "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip",
            "fbms_id_mappings": "https://uni-siegen.sciebo.de/s/w9qWRqML7zSpJDg/download",
            "multicut_uncertainty_code": "https://uni-siegen.sciebo.de/s/aDqq6Zzkm0IimWK/download",
            "multicut_uncertainty_labels": "https://uni-siegen.sciebo.de/s/FQVW9vlmjAWbD3O/download",
            "soft_semantic_segmentation_code": "https://github.com/iyah4888/SIGGRAPH18SSS.git",
            "soft_semantic_segmentation_models": "https://uni-siegen.sciebo.de/s/Sj5Lpu8WqbubNl2/download",
            "pc_pretrain_states": "https://uni-siegen.sciebo.de/s/fgUzpsMO6SspgO1/download"
        }
