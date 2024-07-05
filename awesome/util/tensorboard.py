import logging
from ast import Import
import time as pytime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from awesome.agent.agent import Agent
from awesome.agent.util.metric_entry import MetricEntry

try:
    from torch.utils.tensorboard import SummaryWriter
except (NameError, ImportError, ModuleNotFoundError) as err:
    pass
import json
import logging
import os
import os.path
from pathlib import Path
import torch
import numpy as np

from awesome.event import ModelStepEventArgs
from awesome.event.torch_model_step_event_args import TorchModelStepEventArgs
from awesome.agent.util import Tracker
from awesome.agent.util import LearningMode, LearningScope
from awesome.agent.torch_agent import TorchAgent
from awesome.serialization import ObjectEncoder, JsonConvertible

class Tensorboard():
    """Tensorboard logging adapter with predefined methods.
    """

    @staticmethod
    def check_installed():
        try:
            sm = SummaryWriter
        except NameError:
            raise ImportError(
                "SummaryWriter could not be resolved. Is tensorboard installed?")

    def __init__(self,
                 name: str,
                 logging_directory: str = "./runs/") -> None:
        """Creating a tensorboard logger.

        Parameters
        ----------
        name : str
            The name of the logger, typically corrensponding to the agent.
        logging_directory : str, optional
            The directory where the logs will be inserted to, by default "./runs/"
        """
        Tensorboard.check_installed()
        path = os.path.normpath(os.path.join(logging_directory, name))
        Path(path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Tensorboard logger created at: {path}")
        self.summary_writer = SummaryWriter(path)

    @classmethod
    def for_torch_agent(cls,
                        agent: TorchAgent,
                        log_loss: bool = True,
                        additional_metrics: List[str] = None,
                        logging_directory: str = "./runs/",
                        log_optimizer: bool = True,
                        log_config: bool = True,
                        log_graph: bool = True,
                        log_config_only_once: bool = True,
                        name: Optional[str] = None, 
                        ) -> 'Tensorboard':
        if name is None:
            base_name = os.path.basename(agent.agent_folder)
            folder_name = os.path.dirname(agent.agent_folder)
            name = os.path.basename(base_name)
            logging_directory = folder_name
        logger = Tensorboard(name=name, logging_directory=logging_directory)
        agent.logger = logger
        if log_loss:
            agent.batch_processed.attach(logger.log_loss)
            agent.epoch_processed.attach(logger.log_loss)
        if additional_metrics is not None:
            for metric in additional_metrics:
                agent.batch_processed.attach(logger.log_metric(metric))
                agent.epoch_processed.attach(logger.log_metric(metric))
        if log_optimizer:
            agent.epoch_processed.attach(logger.log_optimizer)
        if log_config:
            agent.epoch_processed.attach(logger.log_config(only_once=log_config_only_once))
        if log_graph:
            agent.batch_processed.attach(logger.log_graph())
        return logger

    def log_loss(self, ctx: Dict[str, Any], output_args: ModelStepEventArgs):
        """Handler for logging the primary loss metric.

        Parameters
        ----------
        ctx : Dict[str, Any]
            The context dict
        output_args : ModelStepEventArgs
            The output args if the model step.
        """
        entry = output_args.tracker.get_recent_performance(scope=LearningScope.to_metric_scope(output_args.scope),
                                                           mode=LearningMode.to_metric_mode(output_args.mode))
        if entry is None:
            return
        tag = Tensorboard.to_tensorboard_tag(entry.tag)
        value = entry.value
        if value is None:
            value = float("inf")
            logging.warning(f"Loss {tag} was None! Setting it to inf.")
        if isinstance(value, complex):
            self.summary_writer.add_scalar(
                tag + "-imag",
                value.imag,
                global_step=entry.global_step,
                walltime=output_args.time)
            value = entry.value.real
        self.summary_writer.add_scalar(
            tag,
            value,
            global_step=entry.global_step,
            walltime=output_args.time)

    def log_metric(self, metric_column: str) -> Callable:
        """Getting a logger function for the given metric name.

        Parameters
        ----------
        metric_column : str
            The metric to log.

        Returns
        -------
        Callable
            The callable event handler method.
        """
        notified = False

        def _log_metric(ctx: Dict[str, Any], output_args: ModelStepEventArgs):
            nonlocal self
            nonlocal notified
            entry = output_args.tracker.get_recent_metric(metric_column,
                                                          scope=LearningScope.to_metric_scope(
                                                              output_args.scope),
                                                          mode=LearningMode.to_metric_mode(output_args.mode))
            if entry is None:
                if not notified:
                    logging.warning(
                        f'Column {metric_column} non in metrics! Can not log to tensorboard.')
                    notified = True
                return
            self.log_metric_entry(entry, time=output_args.time)
        return _log_metric

    def log_metric_entry(self, entry: MetricEntry, time: Optional[float] = None):
        """Logs a metric entry to tensorboard.

        Parameters
        ----------
        entry : MetricEntry
            The metric entry to log.
        time : Optional[float], optional
            The time which should be attached., by default None
        """
        if time is None:
            time = pytime.time()
        tag = Tensorboard.to_tensorboard_tag(entry.tag)
        value = entry.value
        self.log_value(value=value, tag=tag, step=entry.global_step, time=time)
        

    def log_value(self, value: Union[np.ndarray, torch.Tensor, int, float, complex], tag: str, step: int, time: Optional[float] = None):
        """Logs a numeric value to tensorboard.
        This can also be used for logging numpy arrays or tensors.
        If array or tensor is multi dimensional, then the dimensions will be flattened and logged as scalars.
        It will be assumed that the shape of the array or tensor is constant over time, otherwise overlapping occurs

        If the value is a complex number, then the real and imaginary part will be logged as separate scalars.

        Parameters
        ----------
        value : Union[np.ndarray, torch.Tensor, int, float, complex]
            Value to log. Can be a scalar, tensor or numpy array.
        tag : str
            The tag to log to.
        step : int
            The step to log to.
        time : Optional[float], optional
            Optional logging time, by default None
        """
        if isinstance(value, (torch.Tensor, np.ndarray)):
            self._log_vec_type(value=value, tag=tag, step=step, time=time)
        else:
            self._log_scalar(value=value, tag=tag, step=step, time=time)


    def _log_scalar(self, value, tag: str, step: int, time: float):
        """Log method for simple scalars."""
        if isinstance(value, complex):
            self.summary_writer.add_scalar(
                tag + "-imag",
                value.imag,
                global_step=step,
                walltime=time)
            value = value.real
        self.summary_writer.add_scalar(
            tag,
            value,
            global_step=step,
            walltime=time)

    def _log_vec_type(self, value: torch.Tensor, tag: str, step: int, time: float):
        """Multi dimensional log for numpy arrays or tensors"""
        if len(value.shape) > 0:
            # Remove 1 dims
            value = value.squeeze() 
        if len(value.shape) == 0:
            # Proceed as scalar
            self._log_scalar(value=value, tag=tag, step=step, time=time)
        else:
            # Multi dimensional tensor should be logged.
            # Dimensions will be flattend and getting individual steps
            flattened = value.flatten()
            for i, v in enumerate(flattened):
                # Assuming that previous steps had also same size, 
                fl_step = max((step - 1), 0) * len(flattened) + i
                self._log_scalar(value=v, tag=tag, step=fl_step, time=time)


    def log_optimizer(self, ctx: Dict[str, Any], output_args: TorchModelStepEventArgs):
        if output_args.mode == LearningMode.TRAINING:
            params = self._get_optimizer_parameters(output_args.optimizer)
            for k, value in params.items():
                self.summary_writer.add_scalar(
                    k,
                    value,
                    global_step=output_args.tracker.global_epochs,
                    walltime=output_args.time)

    def _log_optimizer(self, optimizer, step: int, time: float):
        params = self._get_optimizer_parameters(optimizer)
        for k, value in params.items():
            self.summary_writer.add_scalar(
                k,
                value,
                global_step=output_args.tracker.global_epochs,
                walltime=output_args.time)

    def _infer_dtype_device(self, model: torch.nn.Module) -> Tuple[torch.dtype, str]:
        for p in model.parameters():
            if isinstance(p, torch.Tensor):
                return p.dtype, p.device
        raise ValueError(
            f"Could not get dtype and device from model {type(model).__name__} as parameters dont contain a tensor!")

    def log_graph(self) -> Callable[[Dict[str, Any], TorchModelStepEventArgs], None]:
        """Returns an executable which can be added to the epoch_processed event.

        Returns
        -------
        Callable[[Dict[str, Any], TorchModelStepEventArgs], None]
            The executable.
        """
        logged = False

        def _log_graph(ctx: Dict[str, Any], output_args: TorchModelStepEventArgs):
            nonlocal self
            nonlocal logged
            if (not logged):
                if output_args.scope != LearningScope.BATCH:
                    logging.warn("The model graph can be only logged in batch mode, as the input is needed!")
                    logged = True
                    return
                if output_args.input is None:
                    logging.warn("Input is not availabe, can not log graph!")
                    logged = True
                    return
                _dt, dev = self._infer_dtype_device(output_args.model)
                _input = output_args.input.to(dtype=_dt, device=dev)
                output_args.model
                self.summary_writer.add_graph(  # TODO Check, throws a AttributeError: 'torch._C.Node' object has no attribute 'cs' here

                    output_args.model,
                    _input
                )
                logged = True
        return _log_graph

    def _format_md_json(self, obj) -> str:
        """Encodes an object to json and formats it for markdown display.

        Parameters
        ----------
        obj : Any
            Any Instance which can be encoded with an ObjectEncoder

        Returns
        -------
        str
            A json formatted string.
        """
        json_str = ObjectEncoder(indent=4, json_convertible_kwargs=dict(
            no_large_data=True, handle_unmatched="jsonpickle")).encode(obj)
        return Tensorboard.json_to_md_format(json_str)

    @staticmethod
    def json_to_md_format(json_str: str) -> str:
        """Formats a json string for markdown display."""
        return "".join("\t" + l for l in json_str.splitlines(True))

    def log_config(self, only_on_change: bool = True, only_once: bool = False) -> Callable[[Dict[str, Any], TorchModelStepEventArgs], None]:
        """Returns a executable for a epoch_processed event which will log the agent config.

        Parameters
        ----------
        only_on_change : bool, optional
            If true only logs the agent config if it has changed., by default True

        only_once : bool, optional
            If set, then the config will be logged only once. If true it will terminate imediatly

        Returns
        -------
        Callable[[Dict[str, Any], TorchModelStepEventArgs], None]
            The executable.
        """
        last_logged: str = None

        def _log(ctx: Dict[str, Any], output_args: ModelStepEventArgs):
            nonlocal last_logged
            nonlocal only_on_change
            nonlocal only_once
            if only_once and last_logged is not None:
                return
            if output_args.mode != LearningMode.TRAINING or output_args.scope != LearningScope.EPOCH:
                return
            agent: Agent
            agent = ctx.get('source')
            args = self._get_config(agent, output_args)
            json_str = self._format_md_json(args)
            if (not only_on_change) or (last_logged is None or json_str != last_logged):
                last_logged = json_str
                self.summary_writer.add_text(f"{type(agent).__name__} config", json_str,
                                             global_step=output_args.tracker.global_epochs)
        return _log

    def _get_config(self, agent: Agent, output_args: ModelStepEventArgs) -> Dict[str, Any]:
        """Gets the config for an agent.

        Parameters
        ----------
        agent : Agent
            _description_
        output_args : ModelStepEventArgs
            _description_

        Returns
        -------
        Dict[str, Any]
            _description_
        """
        torch_args = {}
        if isinstance(output_args, TorchModelStepEventArgs):
            output_args: TorchModelStepEventArgs
            agent: TorchAgent
            torch_args = dict(
                optimizer=type(output_args.optimizer).__name__,
                optimizer_init_args=agent.optimizer_args,
                optimizer_params=self._get_optimizer_parameters(output_args.optimizer),
                loss_args=agent.loss,
            )
        return dict(
            agent_name=agent.name,
            model=type(output_args.model).__name__,
            model_init_args=output_args.model_args,
            loss=output_args.loss_name,
            dataset_config={k: v for k, v in output_args.dataset_config.items() if 'indices' not in k.lower()},
            **torch_args
        )

    def _get_optimizer_parameters(self, optimizer: torch.optim.Optimizer, prefix: str = None) -> Dict[str, Any]:
        """Extract the optimizer parameters into a tag-dict which can be logged.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer
        prefix : str, optional
            A prefix tag, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary of optimizer values.
        """
        name = type(optimizer).__name__
        tag = f'{name}/'
        if prefix is not None:
            tag = prefix
        ret = {}
        it = 0
        for group_num, parameter_group in enumerate(optimizer.param_groups):
            t = tag
            if len(optimizer.param_groups) > 1:
                name = parameter_group.get('name', None)
                if name is None or len(name.strip()) == 0:
                    name = f'group_{group_num}'
                else:
                    name = name.strip()
                t = t + name + '-'
            for parameter in parameter_group:
                if parameter == "name":
                    continue
                if parameter == 'params':
                    continue
                if (isinstance(parameter_group[parameter], float) or
                        isinstance(parameter_group[parameter], int)):
                    ret[t + parameter] = parameter_group[parameter]
                elif isinstance(parameter_group[parameter], (tuple, list)):
                    for i, v in enumerate(parameter_group[parameter]):
                        if isinstance(v, float) or isinstance(v, int):
                            ret[t + parameter + '_' + str(i)] = v
        return ret

    @staticmethod
    def to_tensorboard_tag(metric_tag: str) -> str:
        mode, scope, metric_name = Tracker.split_tag(metric_tag)
        return "/".join([metric_name, scope, mode])
