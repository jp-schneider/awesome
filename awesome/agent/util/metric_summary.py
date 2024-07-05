import math
import random
import re
from typing import Any, Callable, Literal, Optional, Set, Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import pandas as pd
from dataclasses import dataclass, field
from .metric_entry import MetricEntry
import numpy as np
from awesome.agent.util.metric_mode import MetricMode
from awesome.agent.util.metric_scope import MetricScope
import torch
import numpy as np
from awesome.util.format import snake_to_upper_camel

def create_metric_df() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=MetricEntry.df_fields())
    df.set_index("step", inplace=True)
    return df

class DoNotSet():
    pass

DO_NOT_SET = DoNotSet()

LABEL_TEXT_PATTERN = r"(?P<number>[0-9]+). (?P<text>.+)"

def random_circle_point(angle: Optional[int] = None,
                        radius: Optional[int] = None,
                        radius_min: int = 10,
                        radius_max: int = 80) -> Tuple[float, float]:
    if angle is None:
        angle = random.randint(0, 360)
    angle = angle % 360
    rad = angle * (math.pi / 180)
    if radius is None:
        radius = random.randint(radius_min, radius_max)
    return (math.cos(rad)*radius, math.sin(rad)*radius)

@dataclass
class MetricSummary():
    """A metric summary contains the data for one metric."""

    tag: str = field(default=None)
    """Tag for the metric. Should something which describes what the metric is."""

    values: pd.DataFrame = field(default_factory=create_metric_df)
    """The values for the current metric."""

    mode: MetricMode = field(default=MetricMode.VALIDATION)
    """The mode for this summary."""

    scope: MetricScope = field(default=MetricScope.EPOCH)
    """The scope for this summary."""

    is_primary: bool = False
    """Marking if this metric is the main metric for training, so the model will be optimized for this."""

    metric_qualname: str = field(default=None)
    """The fully qualifying name of the metric class / loss function. Will be used to compare metrics."""


    @property
    def metric_name(self) -> str:
        """Gets the metric name by splitting the tag.

        Returns
        -------
        str
            Metric name.
        """
        from awesome.agent.util import Tracker
        return Tracker.split_tag(self.tag)[2]

    @property
    def metric_display_name(self) -> str:
        """
        Returns the display name for the metric."""
        return MetricSummary.get_metric_display_name(self.metric_name)

    @classmethod
    def get_metric_display_name(cls, metric_name: str) -> str:
        """Gets the display name for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric as its used internally.

        Returns
        -------
        str
            A display version of the internal name.
        """
        if '_' in metric_name:
            metric_name = snake_to_upper_camel(metric_name, " ")
        return metric_name

    def process_value(self, value: Any) -> Any:
        if torch is not None:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                else:
                    # Convert tensor to numpy array
                    value = value.detach().cpu().numpy()
        if np is not None:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
        return value

    def log(self, step: int, value: Any, global_step: int):
        """Logs the given value in the value dataframe for the given metric.
        Key must already be inside.

        Parameters
        ----------
        step : int
            The current step of the metric / index.
        value : Any
            The actual metric value.
        global_step : int
            The global step.
        """
        err_count = 0
        done = False
        err = None
        # Theres is a weird error appearing setting a iterable value to an empty dataframe, which dissapears after a second try
        while (not done and err_count < 2):
            try:
                self.values.at[step, "value"] = self.process_value(value)
                self.values.at[step, "global_step"] = global_step
                done=True
            except ValueError as err:
                err_count += 1
        if not done:
            raise err

    def extend(self, amount: int):
        """Extends the current values by amount number of elements.
        Assumes, there is a consecutive ordering.

        Parameters
        ----------
        amount : int
            Number of elements to add.
        """
        m = - 1
        if len(self.values) > 0:
            m = max(self.values.index)
        appended = pd.concat([self.values, pd.DataFrame(
            index=np.arange(m + 1, m + 1 + amount))], ignore_index=True)
        self.values = appended

    def trim(self):
        """Removes entries whitch have no value associated.
        """
        self.values = self.values.drop(
            self.values[pd.isna(self.values['value'])].index, axis=0)

    def get_metric_entry(self, step: int) -> Optional[MetricEntry]:
        """Gets the metric entry at the given step.

        Parameters
        ----------
        step : int
            The step where the entry should be taken from.

        Returns
        -------
        Optional[MetricEntry]
            The metric entry at this point or none if it does not exists.
        """
        last = None
        if step in self.values.index:
            last = self.values.loc[step]
        if last is None:
            return None
        return MetricEntry.from_series(last,
                                       additional_data=dict(
                                           step=step,
                                           tag=self.tag,
                                           metric_qualname=self.metric_qualname))

    def plot(self,
             size: float = 5,
             label: str = None,
             ax: Optional[Axes] = None,
             color: Optional[str] = None,
             xlabel: Optional[str] = None,
             yscale: Optional[str] = None,
             ylabel: Optional[str] = None,
             aggregation: Optional[Callable[[
                 pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray]]] = None,
             best_marker: bool = False,
             best_marker_type: Literal['min', 'max'] = 'min',
             random_marker_text_placement: bool = False,
             marker_text_yformat: Optional[str] = None,
             marker_text_xformat: Optional[str] = None,
             last_marker: bool = False,
             ) -> AxesImage:
        """Plots the given metric to a figure.

        Parameters
        ----------
        size : float, optional
            The size of the figure, by default 5
        label : str, optional
            The label of the plot, e.g. for a description, by default None
        ax : Optional[Axes], optional
            An optional existing axis if the values should be plotted on an existing plot, by default None
        color : Optional[str], optional
            Color argument for the plot function.
        xlabel : Optional[str], optional
            The x label which should be set, if None, it will be the metric name. Pass the DO_NOT_SET object to avoid setting., by default None
        yscale : Optional[str], optional
            The scale of the y axis, None will not change the default behavoir, by default None
        ylabel : Optional[str], optional
            The y label which should be set, if None, it will be the metric name. Pass the DO_NOT_SET object to avoid setting., by default None
        aggregation : Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]], optional
            Optional aggregation or filter which gets called with the (x and y) usually global_step and value of the metric summary.
            The return value should be a tuple containing x and y what should be plotted.Default None.
        best_marker : bool, optional
            If true, the best marker will be plotted, by default False
        best_marker_type : Literal['min', 'max'], optional
            The type of the best marker, by default 'min'
        random_marker_text_placement : bool, optional
            If true, the marker text will be placed randomly, by default False
        marker_text_yformat : Optional[str], optional
            The format of the y value in the marker text, by default None
            None means it will be "{:.2f}" if not float(best_y).is_integer() else "{:0d}"
        marker_text_xformat : Optional[str], optional
            The format of the x value in the marker text, by default None
            None means it will be "{:.2f}" if not float(best_x).is_integer() else "{:0d}"
        last_marker : bool, optional
            If true, the last marker will be plotted, by default False
        Returns
        -------
        AxesImage
            The axis image.
        """
        from awesome.agent.util import Tracker
        fig = None
        if ax is not None:
            fig = plt.gcf()
        else:
            nrows = 1
            ncols = 1
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(ncols * size, nrows * size))
        if label is None:
            label = self.tag
        if xlabel is None:
            xlabel = MetricScope.display_name(self.scope)
        if ylabel is None:
            ylabel = self.metric_display_name

        run_number = None
        match = re.fullmatch(LABEL_TEXT_PATTERN, label)
        if match is not None:
            run_number = match.group("number")
        x = self.values['global_step']
        y = self.values['value']
        if aggregation is not None:
            x, y = aggregation(x, y)
        else:
            # Check shape of x and y
            if torch is not None:
                if len(y) > 0 and isinstance(y.iloc[0], torch.Tensor):
                    if len(y.iloc[0].shape) > 1:
                        # Some dangling dimensions
                        y = y.apply(lambda x: x.squeeze())
                    if torch.prod(torch.tensor(y.iloc[0].shape)) > 1:
                        # Multi dim tensor which needs to be plotted.
                        steps_per_entry = y.apply(lambda x: len(
                            x)).max()  # Get longest value series
                        xx, yy = np.meshgrid(
                            np.arange(steps_per_entry), x.to_numpy())
                        new_x = xx.flatten() + (yy.flatten() * steps_per_entry)
                        x = new_x
                        y = torch.stack([v for v in y]).flatten()

        args = dict()
        xyoffset = None
        if best_marker:
            marker_args = dict()
            y_id = None
            if best_marker_type == 'min':
                y_id = y.argmin()
            elif best_marker_type == 'max':
                y_id = y.argmax()
            else:
                raise ValueError(
                    f"Unknown best marker type {best_marker_type}")
            best_x = x[y_id]
            best_y = y[y_id]
            if 'markersize' not in marker_args:
                marker_args['markersize'] = 12
            if 'fillstyle' not in marker_args:
                marker_args['fillstyle'] = 'full'
            if 'marker' not in marker_args:
                marker_args['marker'] = '.'
            if color is not None:
                marker_args['color'] = color
            ax.plot([best_x], [best_y], **marker_args)
            if marker_text_yformat is None:
                marker_text_yformat = "{:.2f}" if not float(best_y).is_integer() else "{:0d}"
            if marker_text_xformat is None:
                marker_text_xformat = "{:.2f}" if not float(best_x).is_integer() else "{:0d}"
            text = f"{marker_text_yformat.format(best_y)} ({marker_text_xformat.format(best_x)})"
            if xyoffset is None:
                xyoffset = random_circle_point(angle=45 if not random_marker_text_placement else None)
            if run_number is not None:
                text = run_number + ". " + text
            ax.annotate(text, xy=(best_x, best_y), xytext=xyoffset,
                        textcoords="offset points",
                        color=color,
                        bbox=dict(facecolor='white', edgecolor=color, pad=2.0),
                        arrowprops=dict(
                            color=color if color else 'black',
                            shrink=0.05,
                            width=0.01,
                            headwidth=0.00
            )
            )
        if last_marker:
            marker_args = dict()
            last_x = x.iloc[len(x) - 1]
            last_y = y.iloc[len(y) - 1]
            if 'markersize' not in marker_args:
                marker_args['markersize'] = 8
            if 'fillstyle' not in marker_args:
                marker_args['fillstyle'] = 'full'
            if 'marker' not in marker_args:
                marker_args['marker'] = 'X'
            if color is not None:
                marker_args['color'] = color
            ax.plot([last_x], [last_y], **marker_args)
            if marker_text_yformat is None:
                marker_text_yformat = "{:.2f}" if not float(last_y).is_integer() else "{:0d}"
            if marker_text_xformat is None:
                marker_text_xformat = "{:.2f}" if not float(last_x).is_integer() else "{:0d}"
            text = f"{marker_text_yformat.format(last_y)} ({marker_text_xformat.format(last_x)})"
            if run_number is not None:
                text = run_number + ". " + text
            if xyoffset is None:
                xyoffset = random_circle_point(angle=45 if not random_marker_text_placement else None)
            ax.annotate(text, xy=(last_x, last_y), xytext=xyoffset,
                        textcoords="offset points",
                        color=color,
                        bbox=dict(facecolor='white', edgecolor=color, pad=2.0),
                        arrowprops=dict(
                            color=color if color else 'black',
                            shrink=0.05,
                            width=0.01,
                            headwidth=0.00
            )
            )
        if color is not None:
            args["color"] = color
        if len(x) > 10:
            ax.plot(x, y, label=label, **args)
        else:
            ax.scatter(x, y, label=label, **args)
        if yscale:
            ax.set_yscale(yscale)
        if ylabel and not isinstance(ylabel, DoNotSet):
            ax.set_ylabel(ylabel)
        if xlabel and not isinstance(xlabel, DoNotSet):
            ax.set_xlabel(xlabel)
        return fig

    def plot_bar(self,
                 model_name: str,
                 width: float = 0.8,
                 size: float = 5,
                 label: str = None,
                 ax: Optional[Axes] = None,
                 color: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 yscale: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 aggregation: Optional[Callable[[
                     pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray]]] = None,
                 best_bar: bool = True,
                 best_bar_type: Literal['min', 'max'] = 'min',
                 xtext_format: Optional[str] = None,
                 ytext_format: Optional[str] = None,
                 last_bar: bool = False,
                 ) -> AxesImage:
        """Plots the given metric to a figure.

        Parameters
        ----------
        size : float, optional
            The size of the figure, by default 5
        label : str, optional
            The label of the plot, e.g. for a description, by default None
        ax : Optional[Axes], optional
            An optional existing axis if the values should be plotted on an existing plot, by default None
        color : Optional[str], optional
            Color argument for the plot function.
        xlabel : Optional[str], optional
            The x label which should be set, if None, it will be the metric name. Pass the DO_NOT_SET object to avoid setting., by default None
        yscale : Optional[str], optional
            The scale of the y axis, None will not change the default behavoir, by default None
        ylabel : Optional[str], optional
            The y label which should be set, if None, it will be the metric name. Pass the DO_NOT_SET object to avoid setting., by default None
        aggregation : Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]], optional
            Optional aggregation or filter which gets called with the (x and y) usually global_step and value of the metric summary.
            The return value should be a tuple containing x and y what should be plotted.Default None.
        best_bar : bool, optional
            If true, the best marker will be plotted, by default False
        best_bar_type : Literal['min', 'max'], optional
            The type of the best marker, by default 'min'
        xtext_format : Optional[str], optional
            The format of the x value in the axis text, by default None
            None means it will be "{:.2f}" if not float(best_x).is_integer() else "{:0d}"
        ytext_format : Optional[str], optional
            The format of the y value in the axis text, by default None
            None means it will be "{:.2f}" if not float(best_y).is_integer() else "{:0d}"
        last_bar : bool, optional
            If true, the last marker will be plotted, by default False
        Returns
        -------
        AxesImage
            The axis image.
        """
        FORMAT_CHANGE_PATTERN = r"\{\:(?P<fmt>[A-z0-9]*(\.)?[0-9]*[A-z]+)\}"
        from awesome.agent.util import Tracker
        fig = None
        if ax is not None:
            fig = plt.gcf()
        else:
            nrows = 1
            ncols = 1
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(ncols * size, nrows * size))
        if label is None:
            label = self.tag
        if xlabel is None:
            xlabel = MetricScope.display_name(self.scope)
        if ylabel is None:
            ylabel = self.metric_display_name

        run_number = None
        match = re.fullmatch(LABEL_TEXT_PATTERN, label)
        if match is not None:
            run_number = match.group("number")
        x = self.values['global_step']
        y = self.values['value']
        if aggregation is not None:
            x, y = aggregation(x, y)
        else:
            # Check shape of x and y
            if torch is not None:
                if len(y) > 0 and isinstance(y.iloc[0], torch.Tensor):
                    if len(y.iloc[0].shape) > 1:
                        # Some dangling dimensions
                        y = y.apply(lambda x: x.squeeze())
                    if torch.prod(torch.tensor(y.iloc[0].shape)) > 1:
                        # Multi dim tensor which needs to be plotted.
                        steps_per_entry = y.apply(lambda x: len(
                            x)).max()  # Get longest value series
                        xx, yy = np.meshgrid(
                            np.arange(steps_per_entry), x.to_numpy())
                        new_x = xx.flatten() + (yy.flatten() * steps_per_entry)
                        x = new_x
                        y = torch.stack([v for v in y]).flatten()

        args = dict()
        if color is not None:
            args["color"] = color
        if best_bar:
            marker_args = dict()
            y_id = None
            if best_bar_type == 'min':
                y_id = y.argmin()
            elif best_bar_type == 'max':
                y_id = y.argmax()
            else:
                raise ValueError(
                    f"Unknown best marker type {best_bar_type}")
            best_x = x[y_id]
            best_y = y[y_id]
            if xtext_format is None:
                xtext_format = "{:.2g}" if not float(best_x).is_integer() else "{:0d}"
            if ytext_format is None:
                ytext_format = "{:.2g}" if not float(best_y).is_integer() else "{:0d}"
            # Converting pattern to old fromatter
            _yfmt = ytext_format
            m = re.fullmatch(FORMAT_CHANGE_PATTERN, _yfmt)
            if m is not None:
                _yfmt = "%" + m.group("fmt")
            else:
                _yfmt = "%" + "g"
            name = model_name + f" ({xtext_format.format(best_x)})*"
            bar_container = ax.bar(name, best_y, width=width, label=label + (" (Best)" if last_bar else ""), **args)
            ax.bar_label(bar_container, fmt=_yfmt)
        if last_bar:
            marker_args = dict()
            last_x = x.iloc[len(x) - 1]
            last_y = y.iloc[len(y) - 1]
            if xtext_format is None:
                xtext_format = "{:.2g}" if not float(last_x).is_integer() else "{:0d}"
            if ytext_format is None:
                ytext_format = "{:.2g}" if not float(last_y).is_integer() else "{:0d}"
            # Converting pattern to old fromatter
            _yfmt = ytext_format
            m = re.fullmatch(FORMAT_CHANGE_PATTERN, _yfmt)
            if m is not None:
                _yfmt = "%" + m.group("fmt")
            else:
                _yfmt = "%" + "g"
            name = model_name + f" ({xtext_format.format(last_x)})"
            bar_container = ax.bar(name, last_y, width=width, label=label + (" (Last)" if best_bar else ""), **args)
            ax.bar_label(bar_container, fmt=_yfmt)
        if color is not None:
            args["color"] = color
        if yscale:
            ax.set_yscale(yscale)
        if ylabel and not isinstance(ylabel, DoNotSet):
            ax.set_ylabel(ylabel)
        if xlabel and not isinstance(xlabel, DoNotSet):
            ax.set_xlabel(xlabel)
        return fig