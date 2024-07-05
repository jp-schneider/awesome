import inspect
import logging
import os
import types
from typing import Any, Dict, List, Literal, Tuple, Type
from awesome.agent.agent import Agent
from awesome.agent.torch_agent import TorchAgent
from awesome.agent.util.learning_mode import LearningMode
from awesome.dataset.awesome_dataset import AwesomeDataset
from awesome.dataset.sisbosi_dataset import SISBOSIDataset
from awesome.dataset.torch_datasource import TorchDataSource
from awesome.event.agent_save_event_args import SaveStage
from awesome.event.torch_model_step_event_args import TorchModelStepEventArgs
from awesome.event.torch_param_altered_event_args import TorchParamAlteredEventArgs
from awesome.event.watchdogs.learning_rate_stop_training_watchdog import LearningRateStopTrainingWatchdog
from awesome.measures.miou import MIOU
from awesome.measures.pixel_accuracy import PixelAccuracy
from awesome.model.abstract_combined_segmentation_module import AbstractCombinedSegmentationModule, PriorMode
from awesome.model.dynamic_param_module import DynamicParamModule
from awesome.model.wrapper_module import WrapperModule
from awesome.run import handles

from awesome.run.awesome_config import AwesomeConfig
from awesome.run.runner import Runner
from awesome.util.prior_cache import PriorCache
from awesome.util.reflection import dynamic_import
from awesome.util.tensorboard import Tensorboard
import torch
import awesome.run.functions as func
from awesome.dataset.sisbosi_dataset import SISBOSIDataset
from awesome.util.torch import TensorUtil, get_weight_normalized_param_groups
from awesome.util.format import parse_type


class AwesomeRunner(Runner):

    config: AwesomeConfig
    """Configuration of the runner."""

    def __init__(self, config: AwesomeConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def get_sisbosi_segmentation_model_args(self, args: Dict[str, Any]) -> Dict[str, Any]:

        mode = args.pop("input", "rgbxy")

        def _calculate_in_chn(input_mode: Literal['rgb', 'rgbxy', 'xy'], train_set: SISBOSIDataset):
            ''' Returns the number of input channels of the network.'''

            dimsxy = train_set.get_xy_dimension()

            if input_mode == 'rgb':
                return 3
            elif input_mode == 'rgbxy':
                return 3 + dimsxy
            elif input_mode == 'xy':
                return dimsxy
            else:
                raise ValueError(f"Invalid input mode: {input_mode}")
        if 'in_chn' not in args:
            args['in_chn'] = _calculate_in_chn(mode, self.dataloader)
        if 'out_chn' not in args:
            classes = self.dataloader.get_number_of_classes()
            if classes == 2 and self.config.use_binary_classification:
                args['out_chn'] = 1
            else:
                args['out_chn'] = classes
        args['in_type'] = mode

        # Pop dtype as it is not supportet
        args.pop('dtype', None)

        return args

    def _setup_prior_dataset_args(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        set_prior_again = False
        if not self.config.segmentation_training_mode == 'single':
            # Segmentaiton model is in 'multi' or 'none' mode
            if self.config.use_prior_model:
                prior_model_type = parse_type(self.config.prior_model_type, parent_type=torch.nn.Module,
                                              instance_type=types.FunctionType, handle_not_a_class="ignore", variable_name="prior_model_type")

                self.prior_model_type = prior_model_type
                _prior_args = dict(self.config.prior_model_args)

                self.prior_model_args = _prior_args

                if not args.get("spatio_temporal", False):
                    # If not spatio temporal, the prior model is a "prior" and will be different for each image
                    args['prior_model_type'] = prior_model_type
                    args['prior_model_args'] = _prior_args
                else:
                    # If spatio temporal, the prior model is a not a real prior => we keep it as we then training with all images from a sequence.
                    args['prior_model_type'] = None
                    args['prior_model_args'] = None
            else:
                # Segmentation model has no prior
                pass
        else:
            # Single image training
            # The full wrapper module is a "prior" and will be different for each image
            args['prior_model_type'] = WrapperModule
            args['prior_model_args'] = dict()
            set_prior_again = True
        return args, set_prior_again

    def build_data_loader(self, **kwargs) -> TorchDataSource:
        dataset_type = parse_type(
            self.config.dataset_type, parent_type=TorchDataSource, variable_name="dataset_type")
        self.dataset_type = dataset_type

        args = dict()
        args.update(self.config.dataset_args)

        dtype = self._get_dtype()

        if dtype is not None:
            args['dtype'] = dtype
        if not 'split_seed' in args:
            args['split_seed'] = self.config.seed

        if ('scribble_percentage' in inspect.signature(dataset_type.__init__).parameters and
                (hasattr(inspect.signature(parse_type(self.config.loss_type)).__init__, "parameters") and
                 'scribble_percentage' in inspect.signature(parse_type(self.config.loss_type)).__init__.parameters)):
            args['scribble_percentage'] = self.config.scribble_percentage

        args, set_prior_again = self._setup_prior_dataset_args(args)

        # If dataloader has a config init argument, pass the config
        if 'config' in inspect.signature(dataset_type.__init__).parameters:
            args['config'] = self.config

        self.dataloader = dataset_type(**args)

        if set_prior_again:
            # In case of SISBOSI, the prior model is set again to avoid cyclic dependecy in the get_model_args on the dataloader
            self.dataloader.__prior_cache__ = PriorCache(
                args['prior_model_type'], self.get_model_args())

    def patch_agent(self, agent: Agent) -> None:
        logging.warning("Patch Agent is currently not fully implemented!")
        self.agent = agent

        # Set dataloader
        self.agent.training_dataset = self.dataloader

    def _get_wrapper_module_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        args['mode'] = self.config.segmentation_training_mode
        args['input_mode'] = "pixel"
        args['prior_arg_mode'] = "xy_c_preattached"
        args['segmentation_module_gets_targets'] = self.config.segmentation_model_gets_targets
        args['use_segmentation_output_inversion'] = self.config.use_segmentation_output_inversion

        if issubclass(self.dataset_type, (SISBOSIDataset, AwesomeDataset)):
            # dataloader returns these as seperate input
            args['prior_arg_mode'] = "param_clean_grid"
            args['input_mode'] = "pixel" if self.config.dataset_args.get(
                "dimension", "3d") == "2d" else "image"

        args["full_prior"] = self.config.segmentation_training_mode == 'single'

        return args

    def get_model_args(self) -> Tuple[Type, Dict[str, Any]]:
        args = dict()
        args['input_mode'] = "pixel"

        if issubclass(self.dataset_type, (SISBOSIDataset, AwesomeDataset)):
            args['input_mode'] = "pixel" if self.config.dataset_args.get(
                "dimension", "3d") == "2d" else "image"

        args['prior_mode'] = PriorMode.PARTIAL
        if self.config.segmentation_training_mode == 'single':
            args['prior_mode'] = PriorMode.FULL

        self.combined_segmentation_module_type = parse_type(
            self.config.combined_segmentation_module_type, parent_type=AbstractCombinedSegmentationModule, variable_name="combined_segmentation_module_type")

        segmentation_model_type = parse_type(self.config.segmentation_model_type,
                                             parent_type=torch.nn.Module,
                                             instance_type=types.FunctionType,
                                             handle_not_a_class="ignore",
                                             variable_name="segmentation_model_type")

        segment_args = dict(
            self.config.segmentation_model_args, dtype=self._get_dtype())
        self.segmentation_model_args = segment_args

        if issubclass(self.dataset_type, (SISBOSIDataset, AwesomeDataset)):
            from awesome.model.cnn_net import CNNNet
            from awesome.model.fc_net import FCNet

            if (isinstance(segmentation_model_type, Type)
                and (issubclass(segmentation_model_type, CNNNet)
                     or issubclass(segmentation_model_type, FCNet))):

                self.segmentation_model_args = self.get_sisbosi_segmentation_model_args(
                    segment_args)

        self.segmentation_model_type = segmentation_model_type

        if isinstance(segmentation_model_type, types.FunctionType):
            segmentation_model = segmentation_model_type(
                **self.segmentation_model_args)
            if not isinstance(segmentation_model, torch.nn.Module):
                raise ValueError(
                    f"If segmentation model type is a function, it must return a pytorch module, but its: {type(segmentation_model).__name__}!")
        else:
            segmentation_model = segmentation_model_type(
                **self.segmentation_model_args)

        args["segmentation_module"] = segmentation_model

        # Load state dict if specified
        if self.config.segmentation_model_state_dict_path is not None:
            _state_dict = torch.load(
                self.config.segmentation_model_state_dict_path)
            segmentation_model.load_state_dict(_state_dict)

        # Prepare prior model
        if self.config.use_prior_model:
            prior_model_type = parse_type(self.config.prior_model_type,
                                          parent_type=torch.nn.Module,
                                          instance_type=types.FunctionType,
                                          handle_not_a_class="ignore",
                                          variable_name="prior_model_type")
            self.prior_model_type = prior_model_type

            _prior_args = dict(self.config.prior_model_args)

            # _prior_args['dtype'] = self._get_dtype()

            self.prior_model_args = _prior_args

            prior_model = self.prior_model_type(**self.prior_model_args)
            args["prior_module"] = prior_model

        if issubclass(self.combined_segmentation_module_type, WrapperModule):
            args = self._get_wrapper_module_args(args)

        if self.config.combined_segmentation_module_args is not None:
            args.update(self.config.combined_segmentation_module_args)

        return args

    def _get_dtype(self) -> torch.dtype:
        if self.config is None:
            return torch.float32
        dtype = self.config.dtype
        if isinstance(dtype, str):
            if not dtype.startswith("torch."):
                raise ValueError(f"Invalid dtype: {dtype}")
            dtype = dynamic_import(dtype)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        return dtype

    def build_agent(self, **kwargs) -> Agent:
        self.model_type = WrapperModule

        self.model_args = self.get_model_args()

        opt_type = parse_type(self.config.optimizer_type,
                              parent_type=torch.optim.Optimizer,
                              variable_name="optimizer_type")
        self.optim_type = opt_type
        self.optim_args = self.config.optimizer_args

        self.loss_type = parse_type(self.config.loss_type,
                                    variable_name="loss_type")

        if ('scribble_percentage' in inspect.signature(self.loss_type.__init__).parameters):
            self.config.loss_args['scribble_percentage'] = self.config.scribble_percentage
        self.loss = self.loss_type(**self.config.loss_args)

        if self.config.output_folder is not None:
            self.config.output_folder = self.config.output_folder.strip(
                "\"").strip("'")
        self.agent = TorchAgent(
            name=self.config.name_experiment,
            model_type=self.model_type,
            model_args=self.model_args,
            optimizer_type=self.optim_type,
            optimizer_args=self.optim_args,
            loss=self.loss,
            training_dataset=self.dataloader,
            device=self.config.device,
            runs_directory=self.config.runs_path,
            agent_directory=self.config.output_folder,
            batch_metrics=[],
            **self.config.agent_args
        )

        def _enforce_convexity(ctx, args: TorchModelStepEventArgs):
            if args.mode == LearningMode.TRAINING:
                args.model.enforce_convexity()
        self.agent.batch_processed.attach(_enforce_convexity)

        if self.config.use_lr_stop_training_watchdog:
            self.lr_wd = LearningRateStopTrainingWatchdog(
                **self.config.lr_stop_training_watchdog_args)
            self.lr_wd.register(self.agent.epoch_processed)

        def _save_image(ctx, args: TorchModelStepEventArgs):
            if hasattr(args, "mode"):
                if args.mode != LearningMode.TRAINING:
                    return
            step = args.tracker.global_epochs

            if step % self.config.plot_indices_during_training_nth_epoch != 0:
                return
            agent = ctx['source']
            indices = self.config.plot_indices_during_training if self.config.plot_indices_during_training is not None else []
            if not isinstance(indices, list):
                indices = [indices]

            for index in indices:
                if index >= len(self.dataloader):
                    continue
                func.save_result(args.model,
                                 agent.model_gets_targets,
                                 self.dataloader,
                                 agent.agent_folder,
                                 index, step,
                                 output_folder=os.path.join(
                                     "output", f"{index:03d}"),
                                 result_folder=os.path.join(
                                     "mask", f"{index:03d}"),
                                 save_raw=self.config.include_unaries_when_saving
                                 )

        self.agent.epoch_processed.attach(_save_image)
        self.agent.training_starts.attach(_save_image)

        if self.config.save_images_after_pretraining:
            self.agent.after_pretrain.attach(handles.get_final_save_handle(
                plot_indices=self.config.plot_final_indices,
                folder_name="prior",
                save_raw=self.config.include_unaries_when_saving,
                compute_crf=self.config.compute_crf_after_pretraining,
            ))

        self.agent.training_finished.attach(handles.get_final_save_handle(
            plot_indices=self.config.plot_final_indices,
            save_raw=self.config.include_unaries_when_saving,
            compute_crf=self.config.compute_crf_after_training,
            only_execute_on=handles.get_on_training_error(False, count_keyboard_interrupt_as_error=False)))

        # Extra penalty

        def _extra_penalty(ctx, args: TorchModelStepEventArgs):
            if args.mode != LearningMode.TRAINING:
                return
            agent = ctx['source']
            if args.tracker.training_epoch >= self.config.extra_penalty_after_n_epochs and hasattr(agent.loss, "extra_penalty") and not agent.loss.extra_penalty:
                agent.loss.extra_penalty = True
                logging.info("Enabling extra penalty in loss.")
                if self.config.use_reduce_lr_in_extra_penalty_hook:
                    for group in agent._optimizer.param_groups:
                        group['lr'] = group['lr'] * \
                            self.config.reduce_lr_in_extra_penalty_hook_factor
                    if self.config.use_lr_on_plateau_scheduler:
                        # Reset
                        from torch.optim.lr_scheduler import ReduceLROnPlateau
                        self.agent.lr_scheduler: ReduceLROnPlateau
                        self.agent.lr_scheduler._reset()
                # Detach this event
                self.agent.epoch_processed.remove(_extra_penalty)

        if self.config.use_extra_penalty_hook:
            self.agent.epoch_processed.attach(_extra_penalty)

        _log_additional_metrics = handles.get_compute_eval_metrics(
            [
                MIOU(average="binary", invert=True,
                     name="ForegroundBinaryMIOU"),
                PixelAccuracy()],
            reduction=["none", 'mean'],
            log_tensorboard=True,
            compute_crf=self.config.compute_crf_with_metrics,
            only_execute_on=handles.callable_or(
                handles.get_only_training_done(),
                handles.get_only_nth_epoch(self.config.compute_metrics_during_training_nth_epoch))
        )
        self.agent.epoch_processed.attach(_log_additional_metrics)
        self.agent.after_pretrain.attach(handles.get_compute_eval_metrics(
            [
                MIOU(average="binary", invert=True,
                     name="ForegroundBinaryMIOU"),
                PixelAccuracy()],
            reduction=["none", 'mean'],
            log_tensorboard=True
        ))
        self.agent.training_starts.attach(handles.get_compute_eval_metrics(
            [
                MIOU(average="binary", invert=True,
                     name="ForegroundBinaryMIOU"),
                PixelAccuracy()],
            reduction=['none', 'mean'],
            compute_crf=self.config.compute_crf_with_metrics,
            log_tensorboard=True
        ))

        _save_handle = handles.get_save_handle(only_execute_on=handles.callable_or(
            handles.get_only_training_done(),
            handles.get_only_nth_epoch(20, also_after_n=True),
            handles.get_on_save_state(SaveStage.PRETRAINING)
        ))
        self.agent.model_saving.attach(_save_handle)

        _prior_save_handle = handles.get_prior_save_handle(only_execute_on=handles.callable_or(
            handles.get_only_training_done(),
            handles.get_only_nth_epoch(20, also_after_n=True),
            handles.get_on_save_state(SaveStage.PRETRAINING)
        ))
        self.agent.model_saving.attach(_prior_save_handle)

        self.logger = Tensorboard.for_torch_agent(
            self.agent,
            log_loss=True,
            logging_directory=os.path.join(self.agent.agent_folder, ".."),
            log_graph=False)

        if self.config.segmentation_model_gets_targets:
            self.agent.model_gets_targets = True

        if self.config.split_params_in_param_groups:
            def split_in_param_groups(module: WrapperModule):
                ret = []
                if module.segmentation_module is not None:
                    params = None
                    param_groups = get_weight_normalized_param_groups(
                        module.segmentation_module, self.config.weight_decay_on_weight_norm_modules, name_prefix="Segmentation_Module")
                    ret.extend(param_groups)
                if module.prior_module is not None:
                    param_groups = get_weight_normalized_param_groups(
                        module.prior_module, self.config.weight_decay_on_weight_norm_modules, name_prefix="Prior_Module")
                    ret.extend(param_groups)
                return ret
            self.agent.optimizer_parameter_func = split_in_param_groups
        else:
            def _get_params(module: WrapperModule):
                param_groups = get_weight_normalized_param_groups(
                    module, self.config.weight_decay_on_weight_norm_modules)
                return param_groups
            self.agent.optimizer_parameter_func = _get_params

        if self.config.segmentation_training_mode == "none":
            # Exclude segmentation model from optimization
            def exclude_segmentation_model(module: WrapperModule):
                # Exclude segmentation model, just adding prior model
                ret = []
                if module.prior_module is not None:
                    param_groups = get_weight_normalized_param_groups(
                        module.prior_module, self.config.weight_decay_on_weight_norm_modules, name_prefix="Prior_Module")
                    ret.extend(param_groups)
                return ret
            self.agent.optimizer_parameter_func = exclude_segmentation_model

        if self.config.use_prior_model and (isinstance(self.prior_model_type, Type) and issubclass(self.prior_model_type, DynamicParamModule)):
            # Attach event to update optimizer params.
            self.agent.model_args['prior_module'].param_altered.attach(
                self._alter_optimizer)

        if self.config.use_lr_on_plateau_scheduler:
            self.agent.use_reduce_lr_on_plateau_scheduler(
                **self.config.lr_on_plateau_scheduler_args)
        elif self.config.use_step_lr_scheduler:
            self.agent.use_step_lr_scheduler(
                **self.config.step_lr_scheduler_args)
        else:
            pass

        self.agent.should_validate_on_epoch = lambda epoch: (
            (epoch % self.config.validation_each_nth_epoch) == 0 or (epoch == self.config.num_epochs - 1))

    def _alter_optimizer(self, ctx, args: TorchParamAlteredEventArgs):
        prior_module = ctx['source']
        optim: torch.optim.Optimizer = self.agent._optimizer
        prior_group = optim.param_groups[1]
        if args.added_params is not None:
            prior_group['params'].extend(list(args.added_params.values()))
        if args.removed_params is not None:
            existing = [id(p) for p in prior_group['params']]
            astr = repr(args)
            for key, param in args.removed_params.items():
                if id(param) in existing:
                    index = existing.index(id(param))
                    existing.pop(index)
                    prior_group['params'].pop(index)

    def build(self, **kwargs) -> None:
        self.build_data_loader()
        self.build_agent()

    def run(self, *args, **kwargs) -> None:
        pass

    def log_config(self) -> None:
        super().log_config()
        # Log config of outer agent
        text = self.__saved_config__
        if text is None:
            return
        text = Tensorboard.json_to_md_format(text)
        self.logger.summary_writer.add_text("Runner Config", text, 0)

        diff_text = self.logger._format_md_json(self.diff_config)
        self.logger.summary_writer.add_text(
            "Runner Config Diff.", diff_text, 0)

    def train(self, *args, **kwargs) -> None:
        self.agent.train(num_epochs=self.config.num_epochs,
                         tqdm=self.config.use_progress_bar, **kwargs)
