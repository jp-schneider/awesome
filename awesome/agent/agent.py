from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import os
from matplotlib.image import AxesImage
import torch
import matplotlib.pyplot as plt
from awesome.agent.util import (Tracker, LearningMode, LearningScope, MetricMode, MetricScope)
from awesome.event.agent_save_event_args import AgentSaveEventArgs
from awesome.event.model_step_event_args import ModelStepEventArgs
from awesome.util.torch import fourier, inverse_fourier
from datetime import datetime
from awesome.event import Event
from awesome.util.torch import VEC_TYPE
from awesome.util.path_tools import open_folder

class Agent(ABC):
    """Abstract agent implementation."""

    def __init__(self,
                 name: Optional[str] = None,
                 tracker: Optional[Tracker] = None,
                 epoch_metrics: Optional[List[Callable[[VEC_TYPE, VEC_TYPE], VEC_TYPE]]] = None,
                 batch_metrics: Optional[List[Callable[[VEC_TYPE, VEC_TYPE], VEC_TYPE]]] = None,
                 agent_directory: Optional[str] = None,
                 runs_directory: Optional[str] = None,
                 created_at: Optional[datetime] = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.name = name if name else type(self).__name__
        self.tracker: Tracker = tracker if tracker else Tracker()
        self.created_at = created_at if created_at else datetime.now().astimezone()
        
        # Events
        self.model_saving = Event[AgentSaveEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs, when the agent tries to save its model,
        which happens at the end of a train, period.
        Shares its context with the other events of this class."""

        self.batch_processed = Event[ModelStepEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs, when a minibatch was passed threw the model.
        Shares its context with the other events of this class."""

        self.epoch_processed = Event[ModelStepEventArgs](
            source=self, context=self.get_shared_event_context())
        """Event which occurs, when a epoch was completed.
        Shares its context with the other events of this class."""

        self.epoch_metrics: List[Callable[[VEC_TYPE, VEC_TYPE], VEC_TYPE]] = epoch_metrics if epoch_metrics else []
        """Additional epoch metrics, invoked with output and label.
        Type should be Callable(output, target) -> result"""

        self.batch_metrics: List[Callable[[VEC_TYPE, VEC_TYPE], VEC_TYPE]] = batch_metrics if batch_metrics else []
        """Additional batch metrics, invoked with output and label.
           Type should be Callable(output, target) -> result"""
        
        if runs_directory is None:
            runs_directory = "./runs/"
        self.runs_directory: str = runs_directory

        if agent_directory is not None:
            self.runs_directory = os.path.dirname(agent_directory)
            self.agent_directory = agent_directory
            os.makedirs(agent_directory, exist_ok=True)
        else:
            self.agent_directory = agent_directory

        self.progress_bar = True
        self._attach_events()

    def get_shared_event_context(self) -> Dict[str, Any]:
        """Returns the shared event context.

        Returns
        -------
        Dict[str, Any]
            Dictionary for the context.
        """
        if not hasattr(self, "__shared_event_context__ ") or self.__shared_event_context__ is None:
            self.__shared_event_context__ = dict()
        return self.__shared_event_context__

    def _attach_events(self):
        """Attaching default event handlers."""
        self.batch_processed.attach(self._compute_additional_metrics_batch)
        self.epoch_processed.attach(self._compute_additional_metrics_epoch)

    def _compute_additional_metrics_batch(self, ctx, args: ModelStepEventArgs):
        if args.scope == LearningScope.BATCH:
            for metric in self.batch_metrics:
                v = metric(args.output, args.label)
                self.tracker.step_metric(Tracker.get_metric_name(
                    metric), v, in_training=args.mode == LearningMode.TRAINING)

    def _compute_additional_metrics_epoch(self, ctx, args: ModelStepEventArgs):
        if args.scope == LearningScope.EPOCH:
            for metric in self.epoch_metrics:
                v = metric(args.output, args.label)
                self.tracker.epoch_metric(Tracker.get_metric_name(
                    metric), v, in_training=args.mode == LearningMode.TRAINING)

    def _create_tracker_metrics_batch(self, train_set, test_set):
        """Creates all registered additional metrics whithin the tracker for the batch.
        """
        for metric in self.batch_metrics:
            if len(train_set) > 0:
                self.tracker.create_metric(
                    metric, is_primary=False, is_epoch=False, is_training=True, handle_exists="ignore")
            if len(test_set) > 0:
                self.tracker.create_metric(
                    metric, is_primary=False, is_epoch=False, is_training=False, handle_exists="ignore")

    def _create_tracker_metrics_epoch(self, train_set, test_set):
        """Creates all registered additional metrics whithin the tracker for the epoch.
        """
        for metric in self.epoch_metrics:
            if len(train_set) > 0:
                self.tracker.create_metric(metric, is_primary=False, is_epoch=True, is_training=True,
                                           handle_exists="ignore")
            if len(test_set) > 0:
                self.tracker.create_metric(metric, is_primary=False, is_epoch=True, is_training=False,
                                           handle_exists="ignore")

    @property
    def date_name(self) -> str:
        """Name with date attached for easier comparison."""
        return self.name + "_" + self.created_at.strftime("%y_%m_%d_%H_%M_%S")



    @property
    def agent_folder(self) -> str:
        """Returns the folder of the agent.

        Returns
        -------
        str
            Folder of the agent.
        """
        if self.agent_directory is not None:
            return self.agent_directory
        path = os.path.join(self.runs_directory, self.date_name)
        os.makedirs(path, exist_ok=True)
        return path

    def open_folder(self) -> None:  
        """Opens the folder of the agent.
        """
        open_folder(self.agent_folder)

    def save(self, is_training_done: bool = False, **kwargs) -> None:
        # TODO Implemement
        pass
