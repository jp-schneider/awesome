from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import pandas as pd
from awesome.util.reflection import class_name

from awesome.agent.util.metric_entry import MetricEntry
from awesome.agent.util.metric_mode import MetricMode
from awesome.agent.util.metric_scope import MetricScope
from awesome.agent.util.metric_summary import MetricSummary
from datetime import datetime
from copy import deepcopy
import inspect


@dataclass
class Tracker():
    """The tracker tracks progress of a agent during training and also its metrics."""

    global_steps: int = 0
    """The global steps for the agent, w.r.t Pytorch thats the total amount of training and test steps."""

    training_steps: int = 0
    """Number of steps (w.r.t Pytorch mini-batches), done during training."""

    test_steps: int = 0
    """Number of steps (w.r.t Pytorch mini-batches), done during testing / evaluation."""

    training_epoch: int = 0
    """The number of epochs (full data traversals) in training mode for the current agent."""

    test_epoch: int = 0
    """The number of epochs (full data traversals) in test / validation mode for the current agent."""

    last_test_epoch_steps: int = 0
    """Total number of steps within the last test epoch."""

    last_train_epoch_steps: int = 0
    """Total number of steps within the last train epoch."""

    metrics: Dict[str, MetricSummary] = field(default_factory=dict)
    """Metrics contains a log of the current metrics."""

    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    """Datetime when racker was created."""

    def copy(self, ignore_metrics: bool = False) -> 'Tracker':
        """Deepcopies the tracker-

        Parameters
        ----------
        ignore_metrics : bool, optional
            If speciefied removes the metrics, by default False

        Returns
        -------
        Tracker
            A Deepcopied tracker.
        """
        cpy = deepcopy(self)
        if ignore_metrics:
            cpy.metrics = dict()
        return cpy

    @staticmethod
    def get_metric_name(metric: Callable[[Any], Any]) -> str:
        """Gets a name for the metric. Checks whether metric has
        a get_name method and calls it or uses its instance name.

        Parameters
        ----------
        metric : Callable[[Any], Any]
            The metric

        Returns
        -------
        str
            The name for the metric
        """
        if hasattr(metric, 'get_name'):
            name = metric.get_name()
        else:
            if inspect.isfunction(metric) and hasattr(metric, "__name__"):
                # Assume the callable is a fucntion
                name = metric.__name__
            else:
                # Assume the callable is a class
                name = type(metric).__name__
        return name

    def step(self, in_training: bool = False) -> None:
        """Counting a step for the tracker either in training or test.

        Parameters
        ----------
        in_training : bool, optional
            If the step occured in training or test, by default False
        """
        self.global_steps += 1
        if in_training:
            self.training_steps += 1
            self.last_train_epoch_steps += 1
        else:
            self.test_steps += 1
            self.last_test_epoch_steps += 1

    def epoch(self, in_training: bool = False) -> None:
        """Counting the epochs in train / test.
        Rests the last epoch steps counter.

        Parameters
        ----------
        in_training : bool, optional
            If the agent is in training, by default False
        """
        if in_training:
            self.training_epoch += 1
            self.last_test_epoch_steps = 0
        else:
            self.test_epoch += 1
            self.last_train_epoch_steps = 0

    def get_state(self, scope: MetricScope, mode: MetricMode) -> int:
        """Returns the current state for the scope and mode.

        Parameters
        ----------
        scope : MetricScope
            The scope to consider.
        mode : MetricMode
            The mode to consider.

        Returns
        -------
        int
            The local state.

        Raises
        ------
        ValueError
            If scope is invalid.
        """
        if scope == MetricScope.EPOCH:
            return self.training_epoch if mode == MetricMode.TRAINING else self.test_epoch
        elif scope == MetricScope.BATCH:
            return self.training_steps if mode == MetricMode.TRAINING else self.test_steps
        raise ValueError(f"Unknown value for: {scope}")

    def get_global_state(self, scope: MetricScope) -> int:
        """Returns the current state for the scope and mode.

        Parameters
        ----------
        scope : MetricScope
            The scope to consider.

        Returns
        -------
        int
            The local state.

        Raises
        ------
        ValueError
            If scope is invalid.
        """
        if scope == MetricScope.EPOCH:
            return self.global_epochs
        elif scope == MetricScope.BATCH:
            return self.global_steps
        raise ValueError(f"Unknown value for: {scope}")

    def step_metric(self, metric_name: str, value: Any, in_training: bool,
                    is_primary: bool = False, step: Optional[int] = None):
        """Logs a metric in step scope (batch) to the metric summary. Requires that the extend call for 
        this metric must be called in advance to increase performance.

        Parameters
        ----------
        metric_name : str
            Name of the metric, must be unique.
        value : Any
            Actual metric value. Should be a numerical python type.
        in_training : bool
            If the metric is recorded in Training
        is_primary : bool, optional
            If the metric is the primary training metric, by default False
        step : Optional[int], optional
            A step value when a diffrent than the current should be used, by default None
        """
        if step is None:
            step = self.get_steps(in_training=in_training)
        tag = Tracker.assemble_tag(
            metric_name=metric_name, in_training=in_training, is_epoch=False)
        metric_summary = self.metrics[tag]
        metric_summary.log(step=step, value=value,
                           global_step=self.global_steps)

    def epoch_metric(self, metric_name: str, value: Any, in_training: bool,
                     is_primary: bool = False, step: Optional[int] = None):
        """Logs a metric in epoch scope to the metric summary. Requires that the extend call for 
        this metric must be called in advance to increase performance.

        Parameters
        ----------
        metric_name : str
            Name of the metric, must be unique.
        value : Any
            Actual metric value. Should be a numerical python type.
        in_training : bool
            If the metric is recorded in Training
        is_primary : bool, optional
            If the metric is the primary training metric, by default False
        step : Optional[int], optional
            A step value when a diffrent than the current should be used, by default None
        """
        if step is None:
            step = self.get_epoch(in_training=in_training)
        tag = Tracker.assemble_tag(
            metric_name=metric_name, in_training=in_training, is_epoch=True)
        metric_summary = self.metrics[tag]
        metric_summary.log(step=step, value=value,
                           global_step=self.global_epochs)

    def create_epoch_metric(self, metric: Callable[[Any], Any],
                            is_primary: bool = False,
                            handle_exists: Literal['ignore', 'raise'] = "raise") -> Tuple[MetricSummary, MetricSummary]:
        """Create a metric which can be invoked on a per epoch level.

        Parameters
        ----------
        metric : Callable[[Any], Any]
            The callable metric which should be logged.
        is_primary : bool, optional
            If this is the primary metric, by default False

        Returns
        -------
        Tuple[MetricSummary, MetricSummary]
            The created metric.

        Raises
        ------
        ValueError
            When the metric already exists!
        """
        train_summary = self.create_metric(metric, is_primary, True, True, handle_exists=handle_exists)
        test_summary = self.create_metric(metric, is_primary, True, False, handle_exists=handle_exists)
        return train_summary, test_summary

    def create_step_metric(self,
                           metric: Callable[[Any], Any],
                           is_primary: bool = False,
                           handle_exists: Literal['ignore', 'raise'] = "raise") -> Tuple[MetricSummary]:
        """Create a metric which can be invoked on a per step level.

        Parameters
        ----------
        metric : Callable[[Any], Any]
            The metric which should be logged.
        is_primary : bool, optional
            If this is the primary metric, by default False
        handle_exists: Literal['ignore', 'raise'], optional
            How to handle if the metric exists, `raise` 
            will raise an Value error, 'ignore' will 
            do nothing and just return, be default 'raise'
        Returns
        -------
        Tuple[MetricSummary, MetricSummary]
            The created or forwarded metric. Returns the train, validation metric.

        Raises
        ------
        ValueError
            When the metric already exists!
        """
        train_summary = self.create_metric(metric, is_primary, False, True, handle_exists=handle_exists)
        test_summary = self.create_metric(metric, is_primary, False, False, handle_exists=handle_exists)
        return train_summary, test_summary

    def create_metric(self, metric: Callable[[Any], Any],
                      is_primary: bool,
                      is_epoch: bool,
                      is_training: bool,
                      handle_exists: Literal['ignore', 'raise'] = "raise",
                      prefix: str = "") -> MetricSummary:
        """Creates a metric summary for the given metric an parameters.

        Parameters
        ----------
        metric : Callable[[Any], Any]
            The actuall callable metric to get a summary for.
        is_primary : bool
            If the metric is a primary (training / fitting) metric where the system optimizes for.
        is_epoch : bool
            If the metric should be created for an epoch or step (batch) scope.
        is_training : bool
            If the metric should be created for training, or validation purposes.
        handle_exists : Literal[&#39;ignore&#39;, &#39;raise&#39;], optional
            How to handle if the metric exists, `raise` 
            will raise an Value error, 'ignore' will 
            do nothing and just return, by default "raise"

        Returns
        -------
        MetricSummary
            The created / or forwarded metric summary.
        """
        summary = None
        metric_name = prefix + Tracker.get_metric_name(metric)
        tag = Tracker.assemble_tag(
            metric_name=metric_name, in_training=is_training, is_epoch=is_epoch)
        if tag not in self.metrics:
            # Check duplicate primary
            if is_primary:
                primaries = [k for x, k in self.metrics.items() if k.is_primary
                             and k.mode == (MetricMode.VALIDATION if not is_training else MetricMode.TRAINING)
                             and k.scope == (MetricScope.BATCH if not is_epoch else MetricScope.EPOCH)]
                if len(primaries) > 0:
                    raise ValueError(f"There is already an existing metric"
                                     + f"which is primary ({', '.join([k.tag for k in primaries])})")
            summary = MetricSummary(tag=tag, is_primary=is_primary,
                                    scope=(MetricScope.BATCH if not is_epoch else MetricScope.EPOCH),
                                    mode=(MetricMode.VALIDATION if not is_training else MetricMode.TRAINING),
                                    metric_qualname=class_name(metric))
            self.metrics[tag] = summary
        else:
            if handle_exists == "raise":
                raise ValueError(f"Metric: {tag} already exists!")
            else:
                summary = self.metrics[tag]
        return summary

    @staticmethod
    def assemble_tag(metric_name: str, in_training: bool, is_epoch: bool) -> str:
        """Gets the tag for a given metric.
        Will have the form e.g.:
        "train/epoch/MSE"
        or
        "eval/batch/MSE"

        So [in_training]/[is_epoch]/[metric_name]

        Parameters
        ----------
        metric_name : str
            The name of the metric, like MSE etc.
        in_training : bool
            If the metric is recorded in training mode.
        is_epoch : bool
            If this is a epoch or batch metric.

        Returns
        -------
        str
            The assembled metric tag.
        """
        mode = MetricMode.TRAINING.value if in_training else MetricMode.VALIDATION.value
        scope = MetricScope.EPOCH.value if is_epoch else MetricScope.BATCH.value
        return mode + "/" + scope + "/" + metric_name

    def split_tag(tag: str) -> Tuple[str, str, str]:
        """Splits a assembled tag into its
        mode, scope and metric name component.

        Parameters
        ----------
        tag : str
            The assembled tag from assemble_tag

        Returns
        -------
        Tuple[str, str, str]
            Returning (mode, scope, metric_name)
        """
        return tuple(tag.split("/"))

    def get_best_performance(self) -> Optional[MetricEntry]:
        """Gets the best performance which is currently the minimum of the primary metric in a validation epoch.

        Returns
        -------
        Optional[MetricEntry]
            The metric entry if available.
        """
        summary: MetricSummary = self.get_primary_existing_metric(
            scope=MetricScope.EPOCH,
            mode=MetricMode.VALIDATION)
        # TODO Best performance looks currently only on min performance, \
        # this should be dependent from the metric itself.
        min = summary.values[summary.values["value"]
                             == summary.values["value"].min()]
        if len(min) > 0:
            min = min.iloc[0]
        else:
            return None
        return MetricEntry.from_series(min, additional_data=dict(tag=summary.tag))

    def get_recent_performance(self, scope: MetricScope = MetricScope.EPOCH,
                               mode: MetricMode = MetricMode.VALIDATION
                               ) -> Optional[MetricEntry]:
        """Gets the recent performance which is currently the primary metric. Per default this is
          the last validation epoch.

        Parameters
        ----------
        scope : MetricScope, optional
            The scope in which to look at, by default MetricScope.EPOCH
        mode : MetricMode, optional
            The mode for the recent performance, by default MetricMode.VALIDATION

        Returns
        -------
        Optional[MetricEntry]
            The metric entry or None if not available.
        """
        summary: MetricSummary = self.get_primary_existing_metric(
            scope=scope,
            mode=mode)
        if summary is None:
            return None
        # Check if current epoch exists:
        ep = self.get_state(scope=scope, mode=mode)
        sum = summary.get_metric_entry(ep)
        if sum is not None:
            return sum
        # else return the last as recent
        ep = self.get_state(scope=scope, mode=mode) - 1
        return summary.get_metric_entry(ep)

    def get_recent_metric(self, metric_name: str, scope: MetricScope, mode: MetricMode) -> Optional[MetricEntry]:
        """Returns the most recent metric entry for the metric.

        Parameters
        ----------
        metric_name : str
            The name of the metric.
        scope : MetricScope
            The scope where the metric was tracked.
        mode : MetricMode
            The mode where the metric was tracked.

        Returns
        -------
        Optional[MetricEntry]
            The metric entry or none if not found.
        """
        summary = self.get_metric(metric_name=metric_name, scope=scope, mode=mode)
        if summary is None:
            return None
        # Check if current epoch exists:
        ep = self.get_state(scope=scope, mode=mode)
        sum = summary.get_metric_entry(ep)
        if sum is not None:
            return sum
        # else return the last as recent
        ep = self.get_state(scope=scope, mode=mode) - 1
        return summary.get_metric_entry(ep)

    def get_metric(self, metric_name: str, scope: MetricScope, mode: MetricMode) -> Optional[MetricSummary]:
        """Gets a metric with the given args if it exists.

        Parameters
        ----------
        metric_name : str
            The name of the metric.
        scope : MetricScope
            The scope.
        mode : MetricMode
            The mode

        Returns
        -------
        Optional[MetricSummary]
            The found summary or None.
        """
        tag = Tracker.assemble_tag(
            metric_name=metric_name,
            in_training=(mode == MetricMode.TRAINING),
            is_epoch=(scope == MetricScope.EPOCH))
        if tag not in self.metrics:
            return None
        return self.metrics[tag]

    def get_primary_existing_metric(
            self, scope: MetricScope, mode: MetricMode = MetricMode.VALIDATION) -> Optional[MetricSummary]:
        """Returns the primary existing metric.
        Will cover the scenario when no validation takes place, like fitting with a L2 fit.
        In this case it checks if the validation metric doesnt exists, whether the train metric exists.

        Parameters
        ----------
        scope : MetricScope
            The scope of the metric.
        mode : MetricMode, optional
            Its mode, by default MetricMode.VALIDATION

        Returns
        -------
        Optional[MetricSummary]
             A metric summary or None if not found.
        """
        summary: MetricSummary = self.get_primary_metric(
            scope=scope,
            mode=mode)
        if summary is None or len(summary.values) == 0 and (mode == MetricMode.VALIDATION):
            # Loading training if there is no validation
            # Change mode:
            mode = MetricMode.TRAINING
            summary = self.get_primary_metric(scope=scope, mode=mode)
        return summary

    def get_primary_metric(
            self, scope: MetricScope, mode: MetricMode = MetricMode.VALIDATION) -> Optional[MetricSummary]:
        """Gets the primary metric filtered by scope and mode or None if not found.

        Parameters
        ----------
        scope : MetricScope
            The scope where to look at.
        mode : MetricMode, optional
            The metric mode, by default MetricMode.VALIDATION

        Returns
        -------
        Optional[MetricSummary]
            The found summary or none.
        """
        return next((v for x, v in self.metrics.items()
                     if v.is_primary and v.scope == scope and
                     v.mode == mode), None)

    def is_current_state_best_model(self) -> bool:
        """Returning whether the current model state is best in terms of the performance metric."""
        best = self.get_best_performance()
        if best is None:
            return False
        recent = self.get_recent_performance()
        if recent is None or recent.value is None:
            return False
        return recent.value <= best.value

    @property
    def global_epochs(self) -> int:
        """The global epochs which is just the sum of test and validation epochs.

        Returns
        -------
        int
            The number of epochs.
        """
        return self.training_epoch + self.test_epoch

    def get_epoch(self, in_training: bool = False) -> int:
        """Gets the current epoch.

        Parameters
        ----------
        in_training : bool, optional
            If training or eval epochs should be returned, by default False

        Returns
        -------
        int
            The epoch
        """
        if in_training:
            return self.training_epoch
        else:
            return self.test_epoch

    def get_steps(self, in_training: bool = False) -> int:
        """Get the current steps.

        Parameters
        ----------
        in_training : bool, optional
            If training or eval steps should be returned, by default False

        Returns
        -------
        int
            The training steps.
        """
        if in_training:
            return self.training_steps
        else:
            return self.test_steps

    def extend_epoch_metrics(self, amount: int, mode: MetricMode):
        """Extends the epoch metrics to efficiently prolong the dataframes.

        Parameters
        ----------
        amount : int
            The numbers of rows to append.
        batch : MetricMode, optional
            Whether to update training or validation epoch metrics, by default False
        """
        metrics = [x for k, x in self.metrics.items()
                   if x.mode == mode
                   and x.scope == MetricScope.EPOCH]
        for metric in metrics:
            metric.extend(amount=amount)

    def extend_batch_metrics(self, amount: int, mode: MetricMode):
        """Extends the batch metrics to efficiently prolong the dataframes.

        Parameters
        ----------
        amount : int
            The numbers of rows to append.
        batch : MetricMode, optional
            Whether to update training or validation batch metrics, by default False
        """
        metrics = [x for k, x in self.metrics.items()
                   if x.mode == mode
                   and x.scope == MetricScope.BATCH]
        for metric in metrics:
            metric.extend(amount=amount)

    def trim_metrics(self, mode: Optional[MetricMode] = None, scope: Optional[MetricScope] = None):
        """Removes empty rows within the metric summary of all metrics
        with matching mode and scope. If the values are none, there is no filtering on this
        property and all will be trimmed.

        Parameters
        ----------
        mode : Optional[MetricMode]
            The mode of the metric.
        scope : Optional[MetricScope]
            The scope of the metric.
        """
        metrics = [x for k, x in self.metrics.items()
                   if ((x.mode == mode) if mode else True)
                   and ((x.scope == scope) if scope else True)]
        for metric in metrics:
            metric.trim()
