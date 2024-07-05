from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
from awesome.measures.awesome_loss import AwesomeLoss
from awesome.run.config import Config
from dataclasses import dataclass, field
from awesome.run.runner import Runner

from awesome.util.reflection import class_name, dynamic_import


def get_default_lr_on_plateau_scheduler_args() -> Dict[str, Any]:
    return {
        "mode": "min",
        "factor": 0.1,
        "patience": 100,
        "verbose": True,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 100,
        "min_lr": 0,
        "eps": 1e-08
    }


def get_default_step_lr_scheduler_args() -> Dict[str, Any]:
    return {
        "gamma": 0.1,
        "step_size": 100,
        "verbose": True,
    }


def get_default_optim_args() -> Dict[str, Any]:
    return {
        "lr": 0.02,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False
    }


def get_default_lr_stop_training_watchdog_args() -> Dict[str, Any]:
    return {
        "learning_rate": 1e-7,
        "mode": "lte",
        "verbose": True,
    }


def get_default_dataset_args() -> Dict[str, Any]:
    return {
        "dataset_path": "./data/datasets/convexity_dataset",
        "batch_size": 1,
        "split_seed": 42,
        "split_ratio": 1.
    }


def get_default_loss_args() -> Dict[str, Any]:
    return {
        "criterion": torch.nn.BCELoss(),
        "alpha": 1.,
    }


@dataclass
class AwesomeConfig(Config):
    """This is the base config for the awesome segmentation."""

    agent_args: Dict[str, Any] = field(default_factory=dict)
    """Additional init Arguments for the agent. Default is default_factory=dict."""

    dataset_type: Union[Type, str] = field(
        default="awesome.dataset.convexity_segmentation_dataset.ConvexitySegmentationDataset")
    """Type of the dataset. Default is awesome.dataset.convexity_segmentation_dataset.ConvexitySegmentationDataset."""

    dataset_args: Dict[str, Any] = field(
        default_factory=get_default_dataset_args)
    """Arguments of the dataset. Default is default_factory=get_default_dataset_args."""

    combined_segmentation_module_type: Union[Type, str] = field(
        default="awesome.model.wrapper_module.WrapperModule")
    """Type of the combined segmentation module. Default is awesome.model.wrapper_module.WrapperModule."""

    combined_segmentation_module_args: Dict[str, Any] = field(
        default_factory=dict)
    """Arguments of the combined segmentation module, will be addidtionally forwarded when constructing the module. Default is an empty dict."""

    segmentation_model_type: Union[Type, str] = field(
        default="awesome.model.net.Net")
    """Type of the segmentation model. Default is awesome.model.net.Net."""

    segmentation_model_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments of the segmentation model. Default is default_factory=dict."""

    segmentation_training_mode: Literal['multi',
                                        'single', 'none'] = field(default="single")
    """Training mode of the segmentation model. Default is single.
    In single training mode, the segmentation model is individually for each image within the dataset.
    Therefore it needs to be initialized for every image and stored for each image.
    This automatically enables the prior copy over for the segmentation model.
    'multi' means that the segmentation model is trained on all images at once, eg. its parameters staying the same.
    'none' means the segmentation model is not trained at all, e.g. its parameters are excluded from optimization.
    """

    segmentation_model_gets_targets: bool = field(default=False)
    """Whether the segmentation model gets the targets as input during training. Default is False."""

    segmentation_model_state_dict_path: Optional[str] = field(default=None)
    """Path to the segmentation model state dict this will be loaded immdiatly. Default is None."""

    use_segmentation_output_inversion: bool = field(default=False)
    """Whether to invert the segmentation output. Prior usually excepect 0 as fg 1 as bg.
    If model outputs logits for foreground and sigmoid is specified, this can be activated. Default is False."""

    prior_model_type: Union[Type, str] = field(
        default="awesome.model.convex_net.ConvexNet")
    """Type of the convexity model. Default is awesome.model.convex_net.ConvexNet."""

    prior_model_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments of the convexity model. Default is default_factory=dict."""

    use_prior_model: bool = field(default=True)
    """Whether to use the prior model. Default is True."""

    plot_indices_during_training: Optional[List[int]] = field(default=None)

    plot_indices_during_training_nth_epoch: Optional[int] = field(default=100)
    # Plot images during training, each nth (specified) epoch. Default is None.

    compute_metrics_during_training_nth_epoch: Optional[int] = field(
        default=20)
    """How often to compute the metrics during training. Default is 20."""

    compute_crf_with_metrics: bool = field(default=False)
    """Whether to compute the crf with the metrics. Default is False."""

    compute_crf_after_training: bool = field(default=False)
    """Whether to compute the crf after training. Default is False."""

    compute_crf_after_pretraining: bool = field(default=False)
    """Whether to compute the crf after pretraining. For this to work save_images_after_pretraining must also be true. Default is False."""

    save_images_after_pretraining: bool = field(default=False)
    """Whether to all images after pretraining. Default is False."""

    plot_final_indices: Optional[Union[int, List[int]]] = field(default=-1)
    """Plot images after training. Default is -1 meaning all.
    Specify a list of indices to plot only these indices. None would plot None. -1 all."""

    include_unaries_when_saving: bool = field(default=False)
    """Whether save also model output as unaries when generating masks. Default is False."""

    # region Generic Args

    loss_type: Union[Type, str] = field(default=class_name(AwesomeLoss))
    """Type of the loss function. Default is L1."""

    loss_args: Dict[str, Any] = field(default_factory=get_default_loss_args)
    """Arguments of the loss function. Default is get_default_loss_args."""

    use_extra_penalty_hook: bool = field(default=False)
    """Whether to use the extra penalty hook. Default is False."""

    extra_penalty_after_n_epochs: int = field(default=200)
    """At which epoch the extra penalty should be used. Default is 200."""

    use_reduce_lr_in_extra_penalty_hook: bool = field(default=False)
    """Whether to use the reduce lr in extra penalty hook. Default is False."""

    reduce_lr_in_extra_penalty_hook_factor: float = field(default=0.05)

    optimizer_type: Union[Type, str] = field(
        default=class_name(torch.optim.Adam))
    """Type of the optimizer. Default is torch.optim.Adam."""

    optimizer_args: Dict[str, Any] = field(
        default_factory=get_default_optim_args)
    """Arguments of the optimizer. Defaults can be seen in get_default_optim_args."""

    weight_decay_on_weight_norm_modules: float = field(default=5e-5)
    """Weight norm on weight norm modules, if any. Default is 5e-5."""

    split_params_in_param_groups: bool = field(default=False)
    """Whether to split the parameters for segmentation and prior model in param groups.
    Default is True."""

    device: str = field(default="cuda")
    """Device which is used for the training. Default is "cuda"."""

    dtype: str = field(default="torch.float32")
    """Data type which is used for the training data and model arguments. Default is torch.float32."""

    use_lr_on_plateau_scheduler: bool = field(default=False)
    """Whether to use a learning rate scheduler for the optimizer. Default is False."""

    lr_on_plateau_scheduler_args: Dict[str, Any] = field(
        default_factory=get_default_lr_on_plateau_scheduler_args)
    """Arguments of the learning rate scheduler. Defaults can be seen in get_default_lr_on_plateau_scheduler_args."""

    use_step_lr_scheduler: bool = field(default=False)
    """Whether to use a step learning rate scheduler for the optimizer. Default is False."""

    step_lr_scheduler_args: Dict[str, Any] = field(
        default_factory=get_default_step_lr_scheduler_args)
    """Arguments of the learning rate scheduler. Defaults can be seen in get_default_lr_on_plateau_scheduler_args."""

    use_lr_stop_training_watchdog: bool = field(default=True)
    """Whether to use a watchdog which stops the training if the learning rate is too small based on the optimizer in agent. Default is True."""

    lr_stop_training_watchdog_args: Dict[str, Any] = field(
        default_factory=get_default_lr_stop_training_watchdog_args)
    """Arguments of the learning rate watchdog in the agent. Defaults can be seen in get_default_lr_stop_training_watchdog_args."""

    num_epochs: int = field(default=800)
    """Number of epochs. Default is 800."""

    scribble_percentage: float = field(default=0.8)
    """Scribble percentage for the awesome loss. Default is 0.8."""

    use_binary_classification: bool = field(default=True)
    """Will set the output channels to 1 and use a sigmoid activation function for the output. Default is True."""

    validation_each_nth_epoch: int = field(default=100)
    """Validate each nth epoch. Default is 100."""

    # endregion

    # region Semantic Soft Segmentation Specific arguments

    semantic_soft_segmentation_code_dir: str = field(
        default="./third_party/soft_semantic_segmentation")
    """Path to the semantic soft segmentation code directory. Default is ./third_party/soft_semantic_segmentation.
    This is need for pre calculating semantic features."""

    semantic_soft_segmentation_model_checkpoint_dir: str = field(
        default="./data/checkpoints/soft_semantic_segmentation/model")
    """Path to the semantic soft segmentation model checkpoint directory. Default is ./data/checkpoints/soft_semantic_segmentation/model"""

    tf_use_gpu: bool = field(default=False)
    """Whether to use the gpu for the tensorflow code. Default is False."""

    # endregion
    def prepare(self) -> None:
        """Method which gets called after the config is readed from parsed args.
        """
        super().prepare()
        if isinstance(self.dtype, str):
            self.dtype = dynamic_import(self.dtype)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
