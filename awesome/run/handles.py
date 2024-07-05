
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import os

import matplotlib.pyplot as plt
from awesome.agent.util.learning_mode import LearningMode
from awesome.agent.util.learning_scope import LearningScope
from awesome.agent.util.tracker import Tracker
from awesome.dataset.prior_dataset import PriorDataset
from awesome.error.missing_ground_truth_error import MissingGroundTruthError
from awesome.event.agent_save_event_args import SaveStage
from awesome.measures.torch_reducable_metric import TorchReducableMetric
from awesome.model.wrapper_module import WrapperModule
import awesome.run.functions as f
from awesome.event import (TrainingFinishedEventArgs,
                           TorchModelStepEventArgs, TorchAgentSaveEventArgs)
import copy

from awesome.serialization.json_convertible import JsonConvertible
from tqdm.autonotebook import tqdm


def callable_or(*handles: Callable[[Dict[str, Any], Any], bool]) -> Callable[[Dict[str, Any], Any], bool]:
    """Get a handle which executes all handles passed which evaluate to a bool and concatenates them via a bool like or.
    If at least one handle evaluates to a True, the handle will return True.
    If no handles evaluate to a True, the handle will return False.

    May not execute all handles.

    Parameters
    ----------
    handles : Callable[[Dict[str, Any], Any], bool]
        The handles to execute.

    Returns
    -------
    Callable[[Dict[str, Any], Any], bool]
        The handle which executes all handles passed as arguments.
    """
    def _handle_or(ctx: Dict[str, Any], args: Any):
        for handle in handles:
            res = handle(ctx, args)
            if res is True:
                return True
        return False
    return _handle_or


def callable_and(*handles: Callable[[Dict[str, Any], Any], bool]) -> Callable[[Dict[str, Any], Any], bool]:
    """Get a handle which executes all handles passed which evaluate to a bool and concatenates them via a bool like and.
    If at least one handle evaluates to a False, the handle will return False.
    If all handles evaluate to a True, the handle will return True.

    May not execute all handles.

    Parameters
    ----------
    handles : Callable[[Dict[str, Any], Any], bool]
        The handles to execute.

    Returns
    -------
    Callable[[Dict[str, Any], Any], bool]
        The handle which executes all handles passed as arguments.
    """
    def _handle_and(ctx: Dict[str, Any], args: Any):
        for handle in handles:
            res = handle(ctx, args)
            if res is False:
                return False
        return True
    return _handle_and


def get_only_nth_epoch(only_nth_epoch: int = -1, also_after_n: bool = False) -> Callable[[Dict[str, Any],
                                                              Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                                             bool]:
    """Get a handle for checking if a handle should be executed based on the current epoch.
    If only_nth_epoch is -1, the handle will always be executed.
    Otherwise, the handle will only be executed if the current epoch is a multiple of only_nth_epoch.

    Parameters
    ----------
    only_nth_epoch : int, optional
        The multiple of epochs the handle should execute, by default -1

    also_after_n : bool, optional
        If True, the handle will execute after n epochs, by default False
        Even if the epoch is not a multiple of only_nth_epoch, but was not executed for n epochs.

    Returns
    -------
    Callable[[Dict[str, Any], Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]], bool]
        The handle which can be registered on save or step events.
    """    """"""
    last_exec = 0
    def _only_nth_epoch(
            ctx: Dict[str, Any],
            args: Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]) -> bool:
        nonlocal last_exec
        tracker = args.tracker
        if only_nth_epoch == -1:
            should = True
        else:
            should = tracker.training_epoch % only_nth_epoch == 0
        if not should and also_after_n:
            should = tracker.training_epoch > last_exec + only_nth_epoch
        if should:
            last_exec = tracker.training_epoch
        return should
    return _only_nth_epoch

def get_on_save_state(stage: SaveStage) -> Callable[[Dict[str, Any], TorchAgentSaveEventArgs],
                                                             bool]:
    """Get a handle for checking if a handle should be executed based on the save stage.

    Parameters
    ----------

    Returns
    -------
    Callable[[Dict[str, Any], TorchAgentSaveEventArgs], bool]
        The handle which can be registered on save events.
    """ 
    def _only_stage(
            ctx: Dict[str, Any],
            args: TorchAgentSaveEventArgs) -> bool:
        return (args.stage == stage)
    return _only_stage


def get_only_last_epoch() -> Callable[[Dict[str, Any],
                                       Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                      bool]:
    """Get a handle for checking if a handle should be executed based on the current epoch.
    The handle will only be executed if the current epoch is the last epoch.
    If registered on a save event, the handle will only be executed if the training is done.

    Returns
    -------
    Callable[[Dict[str, Any], Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]], bool]
        The handle which can be registered on save or step events.
    """
    def _only_last_epoch(
            ctx: Dict[str, Any],
            args: Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]) -> bool:
        if isinstance(args, TorchAgentSaveEventArgs):
            return args.is_training_done
        else:
            return args.remaining_iterations == 0
    return _only_last_epoch


get_only_training_done = get_only_last_epoch
"""Alias for get_only_last_epoch."""


def get_on_training_error(
        execute_on_error: bool = True,
        count_keyboard_interrupt_as_error: bool = True
) -> Callable[[Dict[str, Any],
               Union[TorchAgentSaveEventArgs, TrainingFinishedEventArgs]],
              bool]:
    """Gets a handle which returns a bool if there was an error on training.
    If execute_on_error is True, then the following handle executes only in case of an error.
    If execute_on_error is False, then the following handle executes only if there was no error.

    Parameters
    ----------
    execute_on_error : bool, optional
        If True, returns True if there is an error within training, if False return True if there was no error, by default True
    count_keyboard_interrupt_as_error : bool, optional
        If True, counts a keyboard interrupt as an error, by default True

    Returns
    -------
    Callable[[Dict[str, Any], Union[TorchAgentSaveEventArgs, TrainingFinishedEventArgs]], bool]
        Handle which can be passed as only_execute_on argument to other handles. 
    """
    def _on_training_error(
            ctx: Dict[str, Any],
            args: Union[TorchAgentSaveEventArgs, TrainingFinishedEventArgs]) -> bool:
        has_err = args.training_error_occurred is not None
        if has_err and isinstance(args.training_error_occurred, KeyboardInterrupt) and not count_keyboard_interrupt_as_error:
            has_err = False
        if execute_on_error:
            return has_err
        else:
            return not has_err
    return _on_training_error


def get_save_handle(
    only_execute_on:
    Optional[Callable[[Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                      bool]] = None) -> Callable[[Dict[str, Any],
                                                  Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                                 None]:
    should_execute = only_execute_on if only_execute_on is not None else (lambda x, y: True)

    def _save_handle(ctx: Dict[str, Any], args: TorchAgentSaveEventArgs):
        if not should_execute(ctx, args):
            return
        agent = ctx['source']
        file_name = "checkpoint_epoch_" + str(args.tracker.global_epochs) + ".pth"
        save_path = os.path.join(agent.agent_folder, file_name)

        args.agent_checkpoint.save(save_path)
        logging.info("Saved checkpoint to " + save_path + " at epoch " + str(args.tracker.global_epochs))
    return _save_handle


def get_prior_save_handle(
    only_execute_on:
    Optional[Callable[[Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                      bool]] = None) -> Callable[[Dict[str, Any],
                                                  Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                                 None]:
    should_execute = only_execute_on if only_execute_on is not None else (lambda x, y: True)

    def _save_handle(ctx: Dict[str, Any], args: TorchAgentSaveEventArgs):
        if not should_execute(ctx, args):
            return
        agent = ctx['source']
        if not (isinstance(agent.training_dataset, PriorDataset) and agent.training_dataset.has_prior):
            return
        file_name = "prior_cache_epoch_" + str(args.tracker.global_epochs) + ".pth"
        save_path = os.path.join(agent.agent_folder, file_name)
        agent.training_dataset.prior_save(save_path)
    return _save_handle


def get_result_save_handle(
    only_execute_on:
    Optional[Callable[[Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                      bool]] = None) -> Callable[[Dict[str, Any],
                                                  Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                                 None]:
    should_execute = only_execute_on if only_execute_on is not None else (lambda x, y: True)

    def _save_handle(ctx: Dict[str, Any], args: TorchAgentSaveEventArgs):
        if not should_execute(ctx, args):
            return
        agent = ctx['source']
        file_name = "result_epoch_" + str(args.tracker.global_epochs) + ".pth"
        save_path = os.path.join(agent.agent_folder, file_name)
        args.agent_result.save(save_path)
    return _save_handle


def get_final_save_handle(
    plot_indices: Optional[Union[int, List[int]]] = None,
    folder_name: str = "final",
    save_raw: bool = False,
    compute_crf: bool = False,
    only_execute_on:
    Optional[Callable[[Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                      bool]] = None) -> Callable[[Dict[str, Any],
                                                  Union[TorchAgentSaveEventArgs, TorchModelStepEventArgs]],
                                                 None]:
    should_execute = only_execute_on if only_execute_on is not None else (lambda x, y: True)

    def _save_handle(ctx: Dict[str, Any], args: TorchAgentSaveEventArgs):
        nonlocal plot_indices
        if not should_execute(ctx, args):
            return
        agent = ctx['source']
        step = agent.tracker.global_epochs

        if plot_indices is None:
            return
        
        # plot all
        if plot_indices == -1:
            for index in tqdm(range(len(agent.training_dataset)), total=len(agent.training_dataset), desc=f"Creating {folder_name} images..."):
                f.save_result(agent._model, 
                              agent.model_gets_targets, 
                              agent.training_dataset, 
                              agent.agent_folder, 
                              index, step, 
                              output_folder=folder_name, 
                              result_folder=f"{folder_name}_mask",
                              save_raw=save_raw,
                              compute_crf=compute_crf
                              )
        else:
            indices = []
            if isinstance(plot_indices, int):
                indices = [plot_indices]
            else:
                indices = plot_indices
            
            for index in tqdm(indices, total=len(indices), desc=f"Creating {folder_name} images..."):
                f.save_result(agent._model, 
                              agent.model_gets_targets, 
                              agent.training_dataset, 
                              agent.agent_folder, 
                              index, 
                              step, 
                              output_folder=folder_name, 
                              result_folder=f"{folder_name}_mask",
                              save_raw=save_raw,
                              compute_crf=compute_crf
                              )
    return _save_handle


def fill_to_equal_shape(val: torch.Tensor, target: torch.Tensor, fill_value: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fills the smaller tensor with a fill_value to match the shape of the bigger tensor.
    Should operate in the batch dimension and expects B x C x H x W tensors.

    Parameters
    ----------
    val : torch.Tensor
        Tensor one.
    target : torch.Tensor
        Tensor two.
    fill_value : float, optional
        The fill value, by default 1

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        val, target
    """
    if val.shape[0] != target.shape[0]:
        if val.shape[0] > target.shape[0]:
            fill_tensor = torch.zeros((val.shape[0] - target.shape[0], *target.shape[1:]), fill_value=fill_value, device=target.device)
            target = torch.cat([target, fill_tensor])
        elif val.shape[0] < target.shape[0]:
            fill_tensor = torch.full((target.shape[0] - val.shape[0], *val.shape[1:]), fill_value=fill_value, device=val.device)
            val = torch.cat([val, fill_tensor])
        else:
            pass
    return val, target

def get_compute_eval_metrics(
    prediction_metrics: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    reduction: Union[str, List[str]] = "none",
    compute_crf: bool = False,
    log_tensorboard: bool = True,
    only_execute_on:
    Optional[Callable[[TorchModelStepEventArgs],
                      bool]] = None) -> Callable[[Dict[str, Any],
                                                  TorchModelStepEventArgs],
                                                 None]:
    should_execute = only_execute_on if only_execute_on is not None else (lambda x, y: True)
    if isinstance(reduction, str):
        reduction = [reduction]

    prior_added = False
    crf_added = False
    has_prior = False
    
    def _compute_metrics(ctx: Dict[str, Any], args: TorchModelStepEventArgs):
        nonlocal prior_added
        nonlocal crf_added
        nonlocal has_prior
        
        if not should_execute(ctx, args):
            return
        agent = ctx['source']
        dataloader = agent.training_dataset
        step = 0
        if hasattr(args, "scope"):
            if args.scope == LearningScope.EPOCH:
                step = agent.tracker.global_epochs
            else:
                step = agent.tracker.global_steps
        else:
            step = agent.tracker.global_epochs
            args.scope = LearningScope.EPOCH

        if not prior_added:
            has_prior = agent._model is not None and isinstance(agent._model, WrapperModule) and agent._model.prior_module is not None
            # If has_prior then copy the metrics to the prior metrics
            if has_prior:
                for metric in list(prediction_metrics):
                    m = copy.deepcopy(metric)
                    m.name = "Prior" + m.get_name()
                    prediction_metrics.append(m)
            prior_added = True

        if compute_crf and not crf_added:
            for metric in list(prediction_metrics):
                if 'prior' in Tracker.get_metric_name(metric).lower():
                    continue
                m = copy.deepcopy(metric)
                m.name = "CRF" + m.get_name()
                prediction_metrics.append(m)
            crf_added = True

        result = dict()
        reduce_metrics = dict()

        for metric in prediction_metrics:
            for red in reduction:
                prefix = red.capitalize() if red != "none" else ""

                name = prefix + Tracker.get_metric_name(metric)
                reduce_metric = reduce_metrics.get(Tracker.get_metric_name(metric), dict())
                reduce_metric[red] = lambda x: TorchReducableMetric.reduce_value(x, reduction=red)
                reduce_metrics[Tracker.get_metric_name(metric)] = reduce_metric

                args.tracker.create_metric(metric, is_primary=False, is_epoch=(
                    args.scope == LearningScope.EPOCH), is_training=False, handle_exists="ignore", prefix=prefix)
        missing_gt_indices = []

        indices = range(len(dataloader))
        total = len(dataloader)
        # If dataloader has
        if hasattr(dataloader, "get_ground_truth_indices"):
            # Saves some time as we don't have to iterate over all indices
            indices = dataloader.get_ground_truth_indices()
            total = len(indices)

        for index in tqdm(indices, total=total, desc="Computing metrics..."):
            try:
                res, ground_truth, img, fg, bg = f.get_result(agent._model, dataloader, index, agent.model_gets_targets, raise_on_missing_ground_truth=True)

            except MissingGroundTruthError as err:
                missing_gt_indices.append(index)
                continue
            
            res = f.split_model_result(res, agent._model, dataloader=dataloader, image=img, compute_crf=compute_crf)
            
            res_pred = res.get("segmentation", None)
            res_prior = res.get("prior", None)

            if ground_truth.squeeze().shape[-2:] != img.shape[1:]:
                ground_truth = ground_truth.squeeze().reshape(img.shape[1:])
    
            # Unsqueeze groudn truth if in shape H x W and others in C x H x W
            if len(ground_truth.shape) == 2 and len(res_pred.shape) == 3:
                ground_truth = ground_truth.unsqueeze(0)

            # If number of objects detected is inequal to gt, then fill with background, either res_pred or gt
            if len(res_pred.shape) == 3:
                res_pred, ground_truth = fill_to_equal_shape(res_pred, ground_truth, fill_value=1.)

            if has_prior and len(res_prior.shape) == 3:
                res_prior, ground_truth = fill_to_equal_shape(res_prior, ground_truth, fill_value=1.)


            for metric in prediction_metrics:
                log = result.get(Tracker.get_metric_name(metric), list())
                result[Tracker.get_metric_name(metric)] = log
                if 'prior' in Tracker.get_metric_name(metric).lower():
                    if has_prior and res_prior is not None:
                        log.append(metric(res_prior.squeeze(), ground_truth.squeeze()))
                elif 'crf' in Tracker.get_metric_name(metric).lower():
                    crf = res.get("segmentation_crf", None)
                    if crf is not None:
                        log.append(metric(crf.squeeze(), ground_truth.squeeze()))
                
                else:
                    log.append(metric(res_pred.squeeze(), ground_truth.squeeze()))


        for metric_name, value in result.items():
            value = torch.stack(value)
            reduce_metric = reduce_metrics[metric_name]
            for red, func in reduce_metric.items():
                prefix = red.capitalize() if red != "none" else ""
                name = prefix + metric_name

                red_value = func(value)
                if args.scope == LearningScope.EPOCH:
                    args.tracker.epoch_metric(name, red_value, False, step=step)
                else:
                    args.tracker.step_metric(name, red_value, False, step=step)
                if log_tensorboard:
                    entry = args.tracker.get_metric(name, LearningScope.to_metric_scope(
                        args.scope), LearningMode.to_metric_mode(LearningMode.VALIDATION)).get_metric_entry(step)
                    agent.logger.log_metric_entry(entry)
    return _compute_metrics
