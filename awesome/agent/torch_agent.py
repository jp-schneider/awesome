from collections import OrderedDict
from datetime import datetime
import inspect
import io
import logging
import os.path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from awesome.agent.torch_agent_checkpoint import TorchAgentCheckpoint
from awesome.dataset.prior_dataset import PriorDataset, PriorManager, create_prior_collate_fn
from awesome.dataset.torch_datasource import TorchDataSource
from awesome.error.stop_training import StopTraining
from awesome.event.agent_save_event_args import SaveStage
from awesome.event.torch_agent_save_event_args import TorchAgentSaveEventArgs
from awesome.event.torch_training_started_event_args import TorchTrainingStartedEventArgs
from awesome.event.training_finished_event_args import TrainingFinishedEventArgs
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.autonotebook import tqdm

from awesome.agent.agent import Agent
from awesome.agent.util import (DataTracker, LearningMode, LearningScope,
                                MetricMode, Tracker)
from awesome.agent.util.metric_scope import MetricScope
from awesome.agent.util.metric_summary import MetricSummary
from awesome.event import Event, TorchModelStepEventArgs
from awesome.event.torch_optimizer_created_event_args import \
    TorchOptimizerCreatedEventArgs
from awesome.measures.tracker_loss import TrackerLoss
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.format import strfdelta
from awesome.util.timer import Timer
from awesome.util.torch import TensorUtil

class TorchAgent(Agent):

    def __init__(self,
                 name: str,
                 model_type: Type,
                 model_args: Dict[str, Any],
                 optimizer_type: Type,
                 optimizer_args: Dict[str, Any],
                 loss: torch.nn.modules.loss,
                 training_dataset: Optional[Dataset] = None,
                 tracker: Optional[Tracker] = None,
                 model_state_dict: Optional[Dict[str, Any]] = None,
                 optimizer_state_dict: Optional[Dict[str, Any]] = None,
                 batch_metrics: List[Callable[[
                     torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 epoch_metrics: List[Callable[[
                     torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 agent_directory: str = None,
                 runs_directory: str = "./runs/",
                 created_at: Optional[datetime] = None,
                 do_pretraining: Optional[bool] = None,
                 pretrain_args: Optional[Dict[str, Any]] = None,
                 pretrain_state_path: Optional[str] = None,
                 force_pretrain: Optional[bool] = None,
                 pretrain_only: Optional[bool] = None,
                 device: str = 'cpu',
                 use_prior_collate_fn: bool = False,
                 **kwargs
                 ):
        super().__init__(name=name,
                         tracker=tracker,
                         batch_metrics=batch_metrics,
                         epoch_metrics=epoch_metrics,
                         agent_directory=agent_directory,
                         runs_directory=runs_directory
                         )

        # Model
        self.model_type: Type = model_type
        self.model_args: Dict[str, Any] = model_args
        self._model: torch.nn.Module = None
        self._model_state_dict: Optional[Dict[str, Any]] = model_state_dict
        """Tracking internal model performance for saving."""

        # Optimizer
        self.optimizer_type: Type = optimizer_type
        self.optimizer_args: Dict[str, Any] = optimizer_args
        self._optimizer: torch.optim.Optimizer = None
        self._optimizer_state_dict: Optional[Dict[str,
                                                  Any]] = optimizer_state_dict
        self.optimizer_parameter_func: Optional[Callable[[
            torch.nn.Module], Dict[str, Any]]] = None
        """Callable function which gets invoked to get the parameters for the optimizer. Can be used to declare param groups etc."""

        self.optimizer_created = Event[TorchOptimizerCreatedEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs, when the optimizer was created.
        Shares its context with the other events of this class."""

        self.after_pretrain = Event[TorchModelStepEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which fires after pretraining if pretraining was performed."""

        self.training_starts = Event[TorchTrainingStartedEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs just before the first epoch is performed."""

        self.training_finished = Event[TorchTrainingStartedEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs after the complete training is finished."""

        # Loss
        self.loss = loss
        self._forward_additional_loss_args = None

        # Data
        self.training_dataset = training_dataset

        self.device = device
        self.batch_progress_bar = True

        self.lr_scheduler = None
        """Optional lr scheduler"""

        # Args for tracking data accross an epoch for the after epoch event.
        self.track_loss = True
        self.track_prediction = False
        self.track_label = False
        self.track_input = False
        self.track_indices = False

        self.__epoch_progress_bar__ = None
        """Epoch progress bar to reuse it."""
        self.__batch_progress_bar__ = None
        """Batch progress bar to reuse it."""
        self.model_gets_targets = False
        """If the model should gets the targets as kwinput in the forward method during training."""

        self.should_validate_on_epoch: Callable[[int], bool] = lambda x: True
        """Executes whether a validation epoch should be performed for the given epoch."""

        self.do_pretraining = do_pretraining
        """Set to true if the model should be pretrained before training. If none checks kwargs of training"""
        self.pretrain_args = pretrain_args if pretrain_args is not None else {}

        self.pretrain_state_path = pretrain_state_path
        """Path to the pretrain state. If set, the pretrain state will be loaded from this path."""

        self.pretrain_only = pretrain_only
        self.force_pretrain = force_pretrain

        self.use_prior_collate_fn = use_prior_collate_fn


    @property
    def forward_additional_loss_args(self) -> bool:
        if self._forward_additional_loss_args is None:
            # Check if loss accepts additional arguments
            spec = inspect.signature(self.loss.__call__).parameters
            if '_input' in spec or 'kwargs' in spec:
                if isinstance(self.loss, torch.nn.Module):
                    spec = inspect.signature(self.loss.forward).parameters
                    self._forward_additional_loss_args = (
                        '_input' in spec or 'kwargs' in spec)
                else:
                    self._forward_additional_loss_args = True
            else:
                self._forward_additional_loss_args = False
        return self._forward_additional_loss_args

    def get_loss_name(self) -> str:
        """Gets a name for the current loss."""
        return Tracker.get_metric_name(self.loss)

    def _get_prepared_model(self) -> torch.nn.Module:
        """Get the model on the correct device.

        Returns
        -------
        torch.nn.Module
            The Learning Model.
        """
        model: torch.nn.Module = self._get_model()
        if not next(model.parameters()).is_cuda and 'cuda' in str(self.device):
            model.to(self.device)
        return model

    def _perform_epoch(self,
                       epoch: int,
                       num_epochs: int,
                       model: Optional[torch.nn.Module],
                       optimizer: Optional[torch.optim.Optimizer],
                       criterion: torch.nn.Module,
                       training_dataset: Dataset,
                       validation_dataset: Dataset,
                       training_batch_size: int,
                       training_data_shuffle: bool,
                       validation_batch_size: int,
                       validation_data_shuffle: bool,
                       shuffle_seed: int,
                       tracker: Tracker,
                       remaining_iterations: int,
                       use_progress_bar: bool,
                       dataset_config: Dict[str, Any],
                       index_in_item: bool = False,
                       batch_progress_bar: Optional[tqdm] = None,
                       epoch_progress_bar: Optional[tqdm] = None,
                       keep_device: bool = False,
                       **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[tqdm]]:
        is_training_finished = False
        try:

            epoch_output_event_args = None
            with Timer() as epoch_timer:
                if epoch_progress_bar:
                    epoch_progress_bar.set_description(
                        'Epoch {}/{}'.format(epoch + 1, num_epochs), refresh=True)

                # If model was removed because of saving
                if model is None:
                    model = self._get_prepared_model()
                    # init optimizer again
                    optimizer = self._get_optimizer(model)

                def worker_init_fn(x):
                    np.random.seed(shuffle_seed)
                    torch.random.manual_seed(shuffle_seed)

                dataloaders = OrderedDict()

                collate_fn = None
                if self.use_prior_collate_fn:
                    use_prior = isinstance(training_dataset, PriorDataset) and training_dataset.has_prior
                    collate_fn = create_prior_collate_fn(has_prior=use_prior)

                # Setting up the data loaders
                if len(training_dataset) > 0:
                    dataloader_train = DataLoader(training_dataset, batch_size=training_batch_size,
                                                  shuffle=training_data_shuffle,
                                                  worker_init_fn=worker_init_fn, collate_fn=collate_fn)
                    dataloaders[LearningMode.TRAINING] = dataloader_train

                if len(validation_dataset) > 0:
                    if self.should_validate_on_epoch is not None and self.should_validate_on_epoch(epoch):
                        dataloader_val = DataLoader(validation_dataset, batch_size=validation_batch_size,
                                                    shuffle=validation_data_shuffle,
                                                    worker_init_fn=worker_init_fn, collate_fn=collate_fn)
                        dataloaders[LearningMode.VALIDATION] = dataloader_val

                # Each epoch has a training and validation phase
                for phase, loader in dataloaders.items():
                    if phase == LearningMode.TRAINING:
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    # Enlarge batch metric tracker
                    tracker.extend_batch_metrics(
                        len(loader), LearningMode.to_metric_mode(phase))

                    if use_progress_bar and self.batch_progress_bar:
                        desc = f'Model-Epoch: {tracker.get_epoch(phase == LearningMode.TRAINING)} / {phase} - Batches'
                        if batch_progress_bar is None:
                            batch_progress_bar = tqdm(
                                total=len(loader),
                                desc=desc,
                                leave=True)
                        else:
                            batch_progress_bar.reset(total=len(loader))
                            batch_progress_bar.set_description(
                                desc
                            )

                    epoch_data_tracker = DataTracker(
                        track_input=self.track_input,
                        track_label=self.track_label,
                        track_loss=self.track_loss,
                        track_prediction=self.track_prediction,
                        track_indices=index_in_item and self.track_indices
                    )
                    try:
                        for idx, item in enumerate(loader):
                            self._perform_step(
                                index=idx,
                                item=item,
                                phase=phase,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                tracker=tracker,
                                epoch_data_tracker=epoch_data_tracker,
                                dataset_config=dataset_config,
                                remaining_iterations=remaining_iterations,
                                batch_progress_bar=batch_progress_bar,
                                epoch_progress_bar=epoch_progress_bar,
                                index_in_item=index_in_item,
                                **kwargs
                            )
                    except StopTraining as err:
                        is_training_finished = True
                        raise err
                    finally:
                        tracker.epoch(
                                in_training=(phase == LearningMode.TRAINING))

                        epoch_loss = epoch_data_tracker.running_loss
                        # Track epoch main metric
                        tracker.epoch_metric(Tracker.get_metric_name(
                            self.loss), value=epoch_loss,
                            in_training=(phase == LearningMode.TRAINING),
                            is_primary=True)

                        epoch_output_event_args = TorchModelStepEventArgs(
                            model=model,
                            model_args=self.model_args,
                            optimizer=optimizer,
                            mode=phase,
                            label=epoch_data_tracker.combined_labels(),
                            output=epoch_data_tracker.combined_predictions(),
                            input=epoch_data_tracker.combined_inputs(),
                            indices=epoch_data_tracker.combined_indices(),
                            loss=epoch_loss,
                            loss_name=self.get_loss_name(),
                            tracker=tracker,
                            remaining_iterations=remaining_iterations if not is_training_finished else 0,
                            dataset_config=dataset_config,
                            scope=LearningScope.EPOCH
                        )
                        self.epoch_processed.notify(
                            epoch_output_event_args)
                        
                        if use_progress_bar:
                            epoch_progress_bar.set_postfix(
                                loss=epoch_loss, refresh=False)
                        
        except StopTraining as err:
            raise
        finally:
            # Ending epoch timing
            logging.info(
                f'Epoch {epoch + 1} / {num_epochs} time {strfdelta(epoch_timer.duration, "%H:%M:%S")}')

            # Compare validation result with best and save model if current is better.
            if (epoch_output_event_args is not None
                and (epoch_output_event_args.mode == LearningMode.VALIDATION or LearningMode.VALIDATION not in dataloaders.keys())
                    and tracker.is_current_state_best_model()):
                best = tracker.get_recent_performance()

                if not keep_device:
                    model.to('cpu')

                self.save(execution_context=kwargs,
                                 keep_device=keep_device,
                                 stage=SaveStage.BEST,
                                 )
                logging.info(f'Accuracy: {best.value}')

                if not keep_device and self.device != "cpu":
                    # Remove reference model optimizer, because its invalid now
                    model = None
                    self._free_optimizer()
                    optimizer = None
        return model, optimizer, batch_progress_bar

    def _decompose_training_item(self, item: Any) -> Tuple[Any, Any, torch.Tensor, Optional[Any]]:
        """Unpacks the item from the training dataset.

        Parameters
        ----------
        item : Any
            The item from the training dataset.

        Returns
        -------
        Tuple[Any, Any, torch.Tensor, Optional[Any]]
            The unpacked item.
            1. The inputs
            2. The labels
            3. The indices
            4. The prior state
        """
        return type(self).decompose_training_item(item, training_dataset=self.training_dataset, use_prior_collate_fn=self.use_prior_collate_fn)

    @classmethod
    def decompose_training_item(cls, item: Any, 
                                training_dataset: TorchDataSource,
                                use_prior_collate_fn: bool = False
                                ) -> Tuple[Any, Any, torch.Tensor, Optional[Any]]:
        """Unpacks the item from the training dataset.

        Parameters
        ----------
        item : Any
            The item from the training dataset.
        training_dataset : TorchDataSource
            The training dataset.

        Returns
        -------
        Tuple[Any, Any, torch.Tensor, Optional[Any]]
            The unpacked item.
            1. The inputs
            2. The labels
            3. The indices
            4. The prior state
        """
        prior_state = None
        # Extraxct prior of attached
        if isinstance(training_dataset, PriorDataset) and training_dataset.has_prior:
            # If its a prior dataset the return will be a tuple (prior, usual_return) of the dataset
            prior_state = item[0]
            key, state = prior_state
            key = key.item()  # Converting it to int again

            if not use_prior_collate_fn:
                # In older versions we using single batch for our priors
                # Removing batch dimension for each entry in state
                new_state = TensorUtil.apply_deep(state, lambda x: x[0] if isinstance(
                    x, torch.Tensor) and x.shape[0] == 1 else x)
            else:
                # In newer versions we using the prior collate function which is already a list of states.
                new_state = state
            
            prior_state = (key, new_state)
            
            item = item[1]

        inputs, labels = item[0], item[1]
        indices = item[2] if training_dataset.returns_index else None

        return inputs, labels, indices, prior_state

    def _perform_step(self,
                      item: Tuple[torch.Tensor, ...],
                      phase: LearningMode,
                      optimizer: torch.optim.Optimizer,
                      model: torch.nn.Module,
                      criterion: torch.nn.Module,
                      tracker: Tracker,
                      epoch_data_tracker: DataTracker,
                      dataset_config: Dict[str, Any],
                      remaining_iterations: int,
                      batch_progress_bar: Optional[tqdm] = None,
                      epoch_progress_bar: Optional[tqdm] = None,
                      index_in_item: bool = False, **kwargs):
        # Getting the inputs and labels unpacked from what training dataset returns
        inputs, labels, indices, prior_state = self._decompose_training_item(
            item)

        device_inputs: torch.Tensor = TensorUtil.to(inputs, device=self.device)
        device_labels: torch.Tensor = TensorUtil.to(labels, device=self.device)

        higher_opt = kwargs.get("higher_optimization", False)

        if not higher_opt:
            # zero the parameter gradients
            optimizer.zero_grad()

        # Optional statistics of batch
        stats = None

        # forward
        # track history if only in train
        with (torch.set_grad_enabled(phase == LearningMode.TRAINING),
              PriorManager(model,
                           prior_state,
                           getattr(self.training_dataset, "__prior_cache__", None), 
                            model_device=self.device, 
                            training=phase == LearningMode.TRAINING) as prior_manager):

            model_kwargs = {}
            if self.model_gets_targets:
                model_kwargs['targets'] = device_labels

            if isinstance(device_inputs, list):
                # Unpacking as list as dataloader wraps multiple args within a list
                device_outputs: torch.Tensor = model(*device_inputs, **model_kwargs)
            else:
                device_outputs: torch.Tensor = model(
                    device_inputs, **model_kwargs)

            # If the loss function accepts additional arguments, pass them
            if self.forward_additional_loss_args:
                loss: torch.Tensor = criterion(
                    device_outputs, device_labels, _input=device_inputs)
            else:
                loss: torch.Tensor = criterion(device_outputs, device_labels)

            if torch.isnan(loss):
                logging.warning("Loss is NaN!")
                breakpoint()
                raise StopTraining()
            # backward + optimize only if in training phase
            if phase == LearningMode.TRAINING:
                if not higher_opt:
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.step(loss)

        # Increase step
        tracker.step(
            in_training=(phase == LearningMode.TRAINING))

        # Getting the loss
        loss_detached = loss.item()
        # detaching tensors
        outputs_detached = TensorUtil.apply_deep(
            device_outputs, fnc=lambda x: x.detach().cpu())

        # Tracking data
        epoch_data_tracker.push(
            loss=loss_detached,
            prediction=outputs_detached,
            label=labels,
            input=inputs,
            indices=indices
        )
        # Track main loss
        tracker.step_metric(Tracker.get_metric_name(
            self.loss),
            value=loss_detached,
            in_training=(phase == LearningMode.TRAINING),
            is_primary=True)

        try:
            # Batch notify
            self.batch_processed.notify(TorchModelStepEventArgs(
                model=model,
                model_args=self.model_args,
                optimizer=optimizer,
                mode=phase,
                label=labels,
                output=outputs_detached,
                indices=indices,
                input=inputs,
                loss=loss_detached,
                loss_name=self.get_loss_name(),
                tracker=tracker,
                remaining_iterations=remaining_iterations,
                dataset_config=dataset_config,
                scope=LearningScope.BATCH
            ))
        finally:
            if batch_progress_bar is not None:
                batch_progress_bar.update()
                batch_progress_bar.set_postfix(
                    loss_avg=epoch_data_tracker.running_loss,
                    loss=loss_detached,
                    refresh=False)

            elif epoch_progress_bar is not None:
                epoch_progress_bar.set_postfix(
                    loss_avg=epoch_data_tracker.running_loss,
                    loss=loss_detached,
                    refresh=False)

    def _pretrain(self,
                  model: torch.nn.Module,
                  train_set: Dataset,
                  test_set: Dataset,
                  use_progress_bar: bool = True,
                  keep_device: bool = True,
                  **kwargs):
        """Method to invoke pretraining and / or loading of pretrain state."""
        if self.do_pretraining == True or (self.do_pretraining is None and kwargs.get("do_pretraining", False)):
            if not isinstance(model, PretrainableModule):
                raise ValueError("Model is not pretrainable!")
            state_loaded = False
            
            pretraining_kwargs = dict(self.pretrain_args)
            pretraining_kwargs.update(kwargs.get("pretrain_args", {}))

            if self.pretrain_state_path is not None:
                # Load state from path
                if os.path.exists(self.pretrain_state_path):
                    try:
                        state = torch.load(
                            self.pretrain_state_path, map_location=self.device)
                        model.pretrain_load_state(
                            train_set=train_set,
                            test_set=test_set,
                            device=self.device,
                            agent=self,
                            use_progress_bar=use_progress_bar,
                            state=state,
                            **pretraining_kwargs)
                        state_loaded = True
                        logging.info(
                            f"Pretrain state loaded from {self.pretrain_state_path}")
                    except Exception as err:
                        logging.error(f"Error loading pretrain state: {err}")
            else:
                # Create path in local directory
                pretrain_path = os.path.join(
                    self.agent_folder, "pretrain_state.pth")
                self.pretrain_state_path = pretrain_path
            if not state_loaded or (self.force_pretrain == True or (self.force_pretrain is None and kwargs.get("force_pretrain", False))):
                logging.info("Starting pretraining...")
                state = model.pretrain(
                    train_set=train_set,
                    test_set=test_set,
                    device=self.device,
                    agent=self,
                    use_progress_bar=use_progress_bar,
                    **pretraining_kwargs)
                # Save the state
                if state is not None:
                    _state = TensorUtil.apply_deep(
                        state, lambda x: x.detach().cpu())
                    os.makedirs(os.path.dirname(self.pretrain_state_path), exist_ok=True)
                    torch.save(_state, self.pretrain_state_path)
                    logging.info(
                        f"Pretrain state saved to {self.pretrain_state_path}")
                else:
                    logging.info("No pretrain state returned, not saving...")
                
            after_pretrain_event_args = TorchModelStepEventArgs(
                model=model,
                model_args=self.model_args,
                mode=LearningMode.TRAINING,
                scope=LearningScope.EPOCH,
                remaining_iterations=0,
                tracker=self.tracker,
            )
            self.after_pretrain.notify(after_pretrain_event_args)
            self.save(execution_context=kwargs,
                                keep_device=keep_device,
                                stage=SaveStage.BEST,
                                )
            
            logging.info("Pretraining done!")

    def train(self, num_epochs=10, keep_device: bool = True, **kwargs):
        training_timer = None
        dataset = self.training_dataset
        if dataset is None:
            raise ValueError(
                f"No dataset was provided, cant train!")

        # Initialize / get the model
        model: torch.nn.Module = self._get_prepared_model()

        # Getting / Creating optimizer
        optimizer = self._get_optimizer(model)

        # The loss function
        criterion = self.loss

        if isinstance(criterion, TrackerLoss):
            criterion: TrackerLoss
            criterion.tracker = self.tracker
            criterion.logger = self.logger
            criterion._recursive_set_name_path()

        # Init data
        train_set, test_set = self._get_test_train_set(dataset)
        dataset_config = dataset.get_config()
        self.dataset_config = dataset_config

        # Init Metric tracker
        tracker = self.tracker
        if len(train_set) > 0:
            tracker.create_metric(
                criterion, is_primary=True, is_epoch=True, is_training=True, handle_exists="ignore")
            tracker.create_metric(
                criterion, is_primary=True, is_epoch=False, is_training=True, handle_exists="ignore")
        if len(test_set) > 0:
            tracker.create_metric(
                criterion, is_primary=True, is_epoch=True, is_training=False, handle_exists="ignore")
            tracker.create_metric(
                criterion, is_primary=True, is_epoch=False, is_training=False, handle_exists="ignore")

        # Main Metric
        self._create_tracker_metrics_batch(train_set, test_set)
        self._create_tracker_metrics_epoch(train_set, test_set)

        tracker.extend_epoch_metrics(num_epochs, MetricMode.TRAINING)
        tracker.extend_epoch_metrics(num_epochs, MetricMode.VALIDATION)

        # Whether to track progress
        use_progress_bar: bool = kwargs.get("tqdm", self.progress_bar)

        # Pretraining if wanted:
        self._pretrain(model, train_set=train_set, test_set=test_set,
                       use_progress_bar=use_progress_bar, keep_device=keep_device, **kwargs)



        if (self.pretrain_only == True or (self.pretrain_only is None and kwargs.get("pretrain_only", False))):
            logging.info("Pretraining done, exiting...")
            return

        epoch_progress_bar: Optional[tqdm] = None
        ep = range(num_epochs)

        if use_progress_bar:
            if self.__epoch_progress_bar__ is not None:
                epoch_progress_bar = self.__epoch_progress_bar__
                epoch_progress_bar.reset(
                    total=len(ep)
                )
            else:
                epoch_progress_bar = tqdm(total=len(ep))
                self.__epoch_progress_bar__ = epoch_progress_bar

        batch_progress_bar: Optional[tqdm] = None
        if use_progress_bar:
            if self.__batch_progress_bar__ is not None:
                batch_progress_bar = self.__batch_progress_bar__

        ts_event_args = TorchTrainingStartedEventArgs(
            model=model,
            model_args=self.model_args,
            optimizer=optimizer,
            loss_name=self.get_loss_name(),
            tracker=tracker,
            remaining_iterations=num_epochs,
            dataset_config=dataset_config,
        )
        self.training_starts.notify(ts_event_args)

        training_err: Exception = None

        try:
            with Timer() as t:
                training_timer = t
                for epoch in ep:
                    stop = False
                    try:
                        model, optimizer, batch_progress_bar = self._perform_epoch(
                            epoch=epoch,
                            num_epochs=num_epochs,
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            training_dataset=train_set,
                            validation_dataset=test_set,
                            training_batch_size=dataset.training_batch_size,
                            training_data_shuffle=dataset.shuffle_in_training_dataloader,
                            validation_batch_size=dataset.validation_batch_size,
                            validation_data_shuffle=dataset.shuffle_in_validation_dataloader,
                            shuffle_seed=dataset.split_seed,
                            tracker=tracker,
                            remaining_iterations=(num_epochs - epoch - 1),
                            use_progress_bar=use_progress_bar,
                            dataset_config=dataset_config,
                            index_in_item=dataset.returns_index,
                            batch_progress_bar=batch_progress_bar,
                            epoch_progress_bar=epoch_progress_bar,
                            keep_device=keep_device,
                            **kwargs
                        )
                    except StopTraining as err:
                        stop = True
                        # Exit
                    finally:
                        if batch_progress_bar is not None:
                            # Manually call a refresh
                            batch_progress_bar.refresh()
                        if epoch_progress_bar is not None:
                            epoch_progress_bar.update()
                        if self.__batch_progress_bar__ is None and batch_progress_bar is not None:
                            self.__batch_progress_bar__ = batch_progress_bar
                    if stop:
                        break
        except (Exception, KeyboardInterrupt) as err:
            logging.error(
                f"Error in training: {type(err).__name__}, Exit gracefully and propagate...")
            training_err = err
            raise
        finally:
            # Trimming metrics.
            tracker.trim_metrics()

            # Moving model to cpu
            if not keep_device and model is not None:
                model.to('cpu')

            self.save(execution_context=kwargs, is_training_done=True,
                      occurred_error=training_err, stage=SaveStage.END)

            if not keep_device and self.device != "cpu":
                # Destrop optimizer, because its invalid now
                self._free_optimizer()
            if batch_progress_bar:
                batch_progress_bar.close()
            logging.info(
                f'Training of agent {self.date_name} complete in {strfdelta(training_timer.duration, "%D days %H:%M:%S")}')
            best = tracker.get_best_performance()
            if best:
                logging.info(
                    f'Best model: {self.name}_Epoch_{best.step} Accuracy: {best.value} Tag: {best.tag}')
            self.training_finished.notify(TrainingFinishedEventArgs(
                training_error_occurred=training_err))

    def _get_model(self) -> torch.nn.Module:
        """Creates the model if it does not exists.

        Returns:
            torch.nn.Module: The model.
        """
        if self._model is None:
            self._model: torch.nn.Module = self.model_type(**self.model_args)
            if self._model_state_dict is not None:
                self._model.load_state_dict(self._model_state_dict)
        return self._model

    def _free_optimizer(self):
        """Saves the state of the optimizer and deletes it.
        """
        if self._optimizer is not None:
            _state = self._optimizer.state_dict()
            self._optimizer_state_dict = _state
            self._optimizer = None

    def _get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Initializes or gets the optimizer. 
        If a state dict is available it will use it.

        Parameters
        ----------
        model : Any
            The model instance where the parameter should be retrieved from.

        Returns
        -------
        torch.optim.Optimizer
            The created optimizer.
        """
        if self._optimizer is None:
            model_params = None
            if self.optimizer_parameter_func is None:
                model_params = model.parameters()
                model_params = list(model_params)
            else:
                model_params = self.optimizer_parameter_func(model)
            self._optimizer: torch.optim.Optimizer = self.optimizer_type(
                model_params, **self.optimizer_args)
            if self._optimizer_state_dict is not None:
                self._optimizer.load_state_dict(self._optimizer_state_dict)
            self.optimizer_created.notify(TorchOptimizerCreatedEventArgs(
                optimizer=self._optimizer, optimizer_args=self.optimizer_args))
        return self._optimizer

    def use_reduce_lr_on_plateau_scheduler(self,
                                           mode: Literal['min', 'max'] = 'min',
                                           factor: float = 0.1,
                                           patience=10,
                                           threshold=1e-4,
                                           threshold_mode: Literal['rel',
                                                                   'abs'] = 'rel',
                                           cooldown: int = 0,
                                           min_lr: float = 0,
                                           eps: float = 1e-8,
                                           verbose=False
                                           ):
        """
            Applies the ReduceLROnPlateau Scheduler on the optimizer when it gets created.
            Need to be called before training.
            Pytorch Doc:

            Reduce learning rate when a metric has stopped improving.
            Models often benefit from reducing the learning rate by a factor
            of 2-10 once learning stagnates. This scheduler reads a metrics
            quantity and if no improvement is seen for a 'patience' number
            of epochs, the learning rate is reduced.

            Args:
                mode (Literal['min', 'max']): One of `min`, `max`. In `min` mode, lr will
                    be reduced when the quantity monitored has stopped
                    decreasing; in `max` mode it will be reduced when the
                    quantity monitored has stopped increasing. Default: 'min'.
                factor (float): Factor by which the learning rate will be
                    reduced. new_lr = lr * factor. Default: 0.1.
                patience (int): Number of epochs with no improvement after
                    which learning rate will be reduced. For example, if
                    `patience = 2`, then we will ignore the first 2 epochs
                    with no improvement, and will only decrease the LR after the
                    3rd epoch if the loss still hasn't improved then.
                    Default: 10.
                threshold (float): Threshold for measuring the new optimum,
                    to only focus on significant changes. Default: 1e-4.
                threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                    dynamic_threshold = best * ( 1 + threshold ) in 'max'
                    mode or best * ( 1 - threshold ) in `min` mode.
                    In `abs` mode, dynamic_threshold = best + threshold in
                    `max` mode or best - threshold in `min` mode. Default: 'rel'.
                cooldown (int): Number of epochs to wait before resuming
                    normal operation after lr has been reduced. Default: 0.
                min_lr (float or list): A scalar or a list of scalars. A
                    lower bound on the learning rate of all param groups
                    or each group respectively. Default: 0.
                eps (float): Minimal decay applied to lr. If the difference
                    between new and old lr is smaller than eps, the update is
                    ignored. Default: 1e-8.
                verbose (bool): If ``True``, prints a message to stdout for
                    each update. Default: ``False``.

            Example:
                >>> # xdoctest: +SKIP
                >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
                >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
                >>> for epoch in range(10):
                >>>     train(...)
                >>>     val_loss = validate(...)
                >>>     # Note that step should be called after validate()
                >>>     scheduler.step(val_loss)
            """
        scheduler_args = dict(
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose
        )

        def _register_lr_plateu_scheduler(ctx: Dict[str, Any], args: TorchOptimizerCreatedEventArgs):
            nonlocal self
            nonlocal scheduler_args
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.lr_scheduler = ReduceLROnPlateau(
                args.optimizer, **scheduler_args)

        self.optimizer_created.attach(_register_lr_plateu_scheduler)

        def _register_lr_plateu_scheduler(ctx: Dict[str, Any], args: TorchModelStepEventArgs):
            # Check if in validation mode or at least where to check for loss improvement
            summary: Optional[MetricSummary] = args.tracker.get_primary_existing_metric(
                scope=MetricScope.EPOCH)
            if summary is not None and (summary.mode == LearningMode.to_metric_mode(args.mode)) and self.lr_scheduler:
                loss = args.loss
                if isinstance(loss, complex):
                    loss = abs(loss)
                self.lr_scheduler.step(loss)
        self.epoch_processed.attach(_register_lr_plateu_scheduler)


    def use_step_lr_scheduler(self,
                                    step_size: int,
                                    gamma: float = 0.1,
                                    last_epoch: int = -1,
                                    verbose=False
                                    ):
        """
        Applies the Step LR Scheduler on the optimizer when it gets created.
        Need to be called before training.
        Pytorch Doc:

        Decays the learning rate of each parameter group by gamma every
        step_size epochs. Notice that such decay can happen simultaneously with
        other changes to the learning rate from outside this scheduler. When
        last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.


        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 60
            >>> # lr = 0.0005   if 60 <= epoch < 90
            >>> # ...
            >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """

        scheduler_args = dict(
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose
        )

        def _register_lr_step_scheduler_optim(ctx: Dict[str, Any], args: TorchOptimizerCreatedEventArgs):
            nonlocal self
            nonlocal scheduler_args
            from torch.optim.lr_scheduler import StepLR
            self.lr_scheduler = StepLR(
                args.optimizer, **scheduler_args)

        self.optimizer_created.attach(_register_lr_step_scheduler_optim)

        def _register_lr_step_scheduler(ctx: Dict[str, Any], args: TorchModelStepEventArgs):
            # Check if in validation mode or at least where to check for loss improvement
            summary: Optional[MetricSummary] = args.tracker.get_primary_existing_metric(
                scope=MetricScope.EPOCH)
            if summary is not None and (summary.mode == LearningMode.to_metric_mode(args.mode)) and self.lr_scheduler:
                self.lr_scheduler.step()
        self.epoch_processed.attach(_register_lr_step_scheduler)

    def _get_test_train_set(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        train, val = dataset.split_indices()
        return Subset(dataset, train), Subset(dataset, val)

    def to_acc(
            self, execution_context: Dict[str, Any] = None, keep_device: bool = False, **kwargs) -> TorchAgentCheckpoint:

        model_state_dict = self._get_model().state_dict()
        optim_state_dict = self._optimizer.state_dict(
        ) if self._optimizer is not None else None
        if keep_device:
            # If the device should be the same, move the state dict tensors to cpu.
            model_state_dict = TensorUtil.apply_deep(
                model_state_dict, lambda x: x.detach().cpu())
            optim_state_dict = TensorUtil.apply_deep(
                optim_state_dict, lambda x: x.detach().cpu())

        clk = TorchAgentCheckpoint(
            name=self.name,
            model_state_dict=model_state_dict,
            model_args=self.model_args,
            model_type=self.model_type,
            optimizer_state_dict=optim_state_dict,
            optimizer_type=self.optimizer_type,
            optimizer_args=self.optimizer_args,
            criterion=self.loss,
            tracker=self.tracker,
            dataset_config=self.training_dataset.get_config(),
            runs_directory=self.runs_directory,
            agent_directory=self.agent_directory,
            created_at=self.created_at,
            execution_context=execution_context,
        )
        return clk

    def save(self,
             execution_context: Dict[str, Any] = None,
             is_training_done: bool = False,
             keep_device: bool = False,
             occurred_error: Optional[Exception] = None,
             stage: SaveStage = SaveStage.UNKNOWN,
             **kwargs
             ) -> TorchAgentSaveEventArgs:
        logging.debug('Saving agent...')
        clk = self.to_acc(execution_context, keep_device=keep_device)
        args = TorchAgentSaveEventArgs(
            name=self.name,
            tracker=self.tracker,
            agent_checkpoint=clk,
            model_type=self.model_type,
            model_args=self.model_args,
            optimizer_type=self.optimizer_type,
            optimizer_args=self.optimizer_args,
            dataset_config=self.dataset_config,
            execution_context=execution_context,
            is_training_done=is_training_done,
            occurred_training_error=occurred_error,
            stage=stage,
        )
        self.model_saving.notify(args)
        return args

    def emergency_save(base_dir: str,
                       module: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       model_input: Any,
                       model_input_kwargs: Any,
                       model_output: Any,
                       targets: Any
                       ):
        os.makedirs(base_dir, exist_ok=True)
        module_state_dict = module.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        module_state_path = os.path.join(base_dir, "module_state_dict.pth")
        torch.save(module_state_dict, module_state_path)
        optimizer_state_path = os.path.join(
            base_dir, "optimizer_state_dict.pth")
        torch.save(optimizer_state_dict, optimizer_state_path)
        model_input_path = os.path.join(base_dir, "model_input.pth")
        torch.save(model_input, model_input_path)
        model_input_kwargs_path = os.path.join(
            base_dir, "model_input_kwargs.pth")
        torch.save(model_input_kwargs, model_input_kwargs_path)
        model_outputs_path = os.path.join(base_dir, "model_outputs.pth")
        torch.save(model_output, model_outputs_path)
        targets_path = os.path.join(base_dir, "targets.pth")
        torch.save(targets, targets_path)
        print(f"Emergency save to {base_dir}")

    @staticmethod
    def from_acc(ckp: TorchAgentCheckpoint) -> 'TorchAgent':
        """Creates the torch agent from a checkpoint instance.

        Parameters
        ----------
        ckp : TorchAgentCheckpoint
            The checkpoint instance.

        Returns
        -------
        TorchAgent
            The created agent.
        """
        agent = TorchAgent(
            name=ckp.name,
            model_state_dict=ckp.model_state_dict,
            model_args=ckp.model_args,
            model_type=ckp.model_type,
            tracker=ckp.tracker,
            optimizer_state_dict=ckp.optimizer_state_dict,
            optimizer_args=ckp.optimizer_args,
            optimizer_type=ckp.optimizer_type,
            loss=ckp.criterion,
            dataset_config=ckp.dataset_config,
            runs_directory=ckp.runs_directory,
            agent_directory=ckp.agent_directory,
            created_at=ckp.created_at,
        )

        return agent

    @classmethod
    def load_acc(cls, file_name_or_buf: Union[os.PathLike[str], io.BytesIO]) -> 'TorchAgentCheckpoint':
        """Loads a checkpoint which can be converted in the instance of the current type.

        Parameters
        ----------
        file_name_or_buf : Union[os.PathLike[str], BytesIO]
            File name or buffer to load from.   

        Returns
        -------
        TorchAgentCheckpoint
            Convertible checkpoint.
        """
        return TorchAgentCheckpoint.load(file_name_or_buf)
