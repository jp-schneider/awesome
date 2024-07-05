
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable
from matplotlib.image import AxesImage
import torch
import matplotlib.pyplot as plt
from awesome.run.functions import saveable
from awesome.serialization import JsonConvertible
import pandas as pd
from enum import Enum
from awesome.util.temporary_property import TemporaryProperty
from awesome.error import ArgumentNoneError
from awesome.analytics.result_model import ResultModel
import math
import numpy as np
from awesome.agent.util import Tracker
from awesome.agent.util.metric_summary import MetricSummary
import pandas as pd
from awesome.util.path_tools import numerated_file_name, open_folder
from datetime import datetime
from pandas.io.formats.style import Styler

class MetricReference(Enum):
    BEST = "best"
    LAST = "last"
    ALL = "all"


class MetricMode(Enum):
    MIN = "min"
    MAX = "max"


def convert_metric_reference(value: Optional[Literal['best', 'last']]) -> MetricReference:
    if value is None:
        return MetricReference.BEST
    if isinstance(value, MetricReference):
        return value
    if isinstance(value, str):
        value = MetricReference(value)
    if isinstance(value, list):
        value = [convert_metric_reference(v) for v in value]
    return value


def convert_metric_mode(value: Optional[Literal['min', 'max']]) -> MetricMode:
    if value is None:
        return MetricMode.MIN
    if isinstance(value, str):
        value = MetricMode(value)
    if isinstance(value, list):
        value = [convert_metric_mode(v) for v in value]
    return value


class ResultComparison():

    models: List[ResultModel]

    numbering: bool
    """If results should be numbered."""

    def __init__(self, models: List[ResultModel], output_folder: str = None, numbering: bool = True):
        if models is None:
            raise ArgumentNoneError("models")
        self.models = models
        self.numbering = numbering
        if output_folder is None:
            output_folder = f"../output/results/{datetime.now().strftime('%y_%m_%d_%H_%M_%S')}"
            os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
        self.assign_numbers()

    def open_folder(self):
        """Opens the output folder of the comparison."""
        open_folder(os.path.normpath(self.output_folder))

    @property
    def numbering(self) -> bool:
        return self.models[0].numbering

    @numbering.setter
    def numbering(self, value: bool):
        for model in self.models:
            model.numbering = value


    def get_important_default_values(self, 
                base_config_path: str,
                additional_default_values: Optional[Dict[str, Any]] = None,
                use_diff_config: bool = True,
                ) -> Dict[str, Dict[str, Any]]:
        """Gets the important default values from a base config file which
        where altered in the different models.
        This can be used to show changes in between the models.

        Parameters
        ----------
        base_config_path : str
            Path to the base config which should be considered.
        additional_default_values : Optional[Dict[str, Any]], optional
            Additional default values which may not be declared within the base
            config as they are defined in the code., by default None

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with:
                - default_values: The default values which are altered in the models.
                - flattend_keys: The keys of the default values in a flattend format.
        """        
        from awesome.util.diff import nested_keys, combine_nested_keys, flatten, filter, changes
        
        base_config = JsonConvertible.load_from_file(base_config_path)
        base_config_dict = base_config.to_json_dict(no_uuid=True, no_large_data=True)

        _all_diff_cfgkeys = None

        if use_diff_config:
            _all_diff_cfgkeys = [nested_keys(x.run_config.diff_config) for x in self.models]
        else:
            _changes = [changes(base_config_dict, x.run_config.to_json_dict(no_uuid=True, no_large_data=True)) for x in self.models]
            _all_diff_cfgkeys = [nested_keys(x) for x in _changes]

        if additional_default_values is not None:
            for key, value in additional_default_values.items():
                base_config_dict[key] = value

        searched_item_keys = combine_nested_keys(_all_diff_cfgkeys)

        default_values = filter(base_config_dict, searched_item_keys)
        
        flat_keys = flatten(searched_item_keys, separator="__", keep_empty=True)
        
        ret = dict(default_values=default_values, flattend_keys=flat_keys)
        return ret


    def get_save_path(self, path_to_file: str, override: bool = False) -> str:
        """Returns a save path within the output directory.
        If override is false it will check for existing files an rename the path eventually.

        Parameters
        ----------
        path_to_file : str
            A subpath within the folder.
        override : bool, optional
            If a file should be overriden, by default False

        Returns
        -------
        str
            The existing path.
        """
        path = os.path.join(self.output_folder, path_to_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not override and not os.path.isdir(path):
            path = numerated_file_name(path)
        return path

    def assign_numbers(self, force: bool = False):
        if force:
            for i in range(0, len(self.models)):
                self.models[i].config.number = i + 1
                self.models[i].save_config()
        else:
            existing = [model.config.number for model in self.models if model.config.number is not None]
            free_numbers = [i for i in range(1, len(self.models) + 1) if i not in existing]
            for model in self.models:
                if model.config.number is None:
                    next_num = free_numbers[0]
                    model.config.number = next_num
                    model.save_config()
                    free_numbers.remove(next_num)
                elif model.config.number not in existing:
                    # Duplicate, reassign number
                    next_num = free_numbers[0]
                    model.config.number = next_num
                    model.save_config()
                    free_numbers.remove(next_num)
                else:
                    # Remove from existing
                    existing.remove(model.config.number)

    def get_number_format(self) -> Optional[str]:
        num_format = None
        if self.numbering:
            max_num = max([x.config.number for x in self.models])
            num_format = f"{{:0{len(str(max_num))}d}}"
        return num_format
    

    def get_trackers(self) -> Dict[str, Tracker]:
        def get_tracker(model) -> Tracker:
            try:
                return model.get_tracker(-1)
            except IndexError as err:
                logging.warning(f"Model: {model.name} dont have a checkpoint!")
        trackers = {model.name: get_tracker(model) for model in self.models}

        return trackers

    def get_tracked_metrics(self) -> Dict[str, List[str]]:
        trackers = self.get_trackers()
        metrics = dict()
        for name, tracker in trackers.items():
            for metric, summary in tracker.metrics.items():
                has_metric = None
                if metric not in metrics:
                    has_metric = list()
                    metrics[metric] = has_metric
                else:
                    has_metric = metrics[metric]
                has_metric.append(name)
        return metrics

    def get_metrics(self, metric_name: str) -> Dict[str, MetricSummary]:
        trackers = self.get_trackers()
        metrics = dict()
        for name, tracker in trackers.items():
            if metric_name in tracker.metrics:
                metrics[name] = tracker.metrics[metric_name]
        return metrics

    @saveable()
    def plot_metric(self,
                    metric_name: str,
                    size: float = 5,
                    top_n: int = -1,
                    top_ref: Literal['min', 'max'] = 'min',
                    top_mode: Literal['best', 'last'] = 'best',
                    ylim: Tuple[float, float] = None,
                    xlim: Tuple[float, float] = None,
                    **kwargs) -> AxesImage:
        import matplotlib
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}

        nrows = 1
        ncols = 1
        figsize = None
        if isinstance(size, tuple):
            figsize = size
        else:
            figsize = (ncols * size, nrows * size)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        # Set default cmap
        cmap_name = "tab10"
        if len(metrics) > 10:
            cmap_name = "tab20"
        cmap = matplotlib.colormaps[cmap_name]

        if top_n > 0:
            def _top_key(name, summary: MetricSummary):
                if top_mode == "best":
                    if top_ref == "min":
                        return summary.values['value'].min()
                    elif top_ref == "max":
                        return summary.values['value'].max()
                    else:
                        raise NotImplementedError()
                elif top_mode == "last":
                    return summary.values.iloc[-1]['value']
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(*x), reverse=(top_ref == "max"))[:top_n])
            metrics = dict(sorted(metrics.items(), key=lambda x: models[x[0]].display_name))

        for i, (model_name, summary) in enumerate(metrics.items()):
            model: ResultModel = models[model_name]
            color = cmap.colors[i % cmap.N]
            try:
                summary.plot(ax=ax, label=model.display_name, color=color, **kwargs)
            except Exception as err:
                logging.exception(f"Could not plot {metric_name} for tracker {model.name}")
        if ylim is not None:
            plt.ylim(*ylim)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.legend()
        plt.grid(axis="y")
        return fig

    @saveable()
    def plot_metric_bar(self,
                        metric_name: str,
                        size: float = 5,
                        top_n: int = -1,
                        top_ref: Literal['min', 'max'] = 'min',
                        top_mode: Literal['best', 'last'] = 'best',
                        order: bool = True,
                        ylim: Tuple[float, float] = None,
                        xlim: Tuple[float, float] = None,
                        **kwargs) -> AxesImage:
        import matplotlib
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}

        nrows = 1
        ncols = 1
        figsize = None
        if isinstance(size, tuple):
            figsize = size
        else:
            figsize = (ncols * size, nrows * size)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        # Set default cmap
        cmap_name = "tab10"
        if len(metrics) > 10:
            cmap_name = "tab20"
        cmap = matplotlib.colormaps[cmap_name]

        def _top_key(name, summary: MetricSummary):
            if top_mode == "best":
                if top_ref == "min":
                    return summary.values['value'].min()
                elif top_ref == "max":
                    return summary.values['value'].max()
                else:
                    raise NotImplementedError()
            elif top_mode == "last":
                return summary.values.iloc[-1]['value']

        if top_n > 0:
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(*x), reverse=(top_ref == "max"))
                           [:top_n])
            if not order:
                metrics = dict(sorted(metrics.items(), key=lambda x: models[x[0]].display_name))
        elif order:
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(*x), reverse=(top_ref == "max")))

        for i, (model_name, summary) in enumerate(metrics.items()):
            model: ResultModel = models[model_name]
            color = cmap.colors[i % cmap.N]
            try:
                summary.plot_bar(model.display_name, ax=ax, label=model.display_name, color=color, **kwargs)
            except Exception as err:
                logging.exception(f"Could not plot {metric_name} for tracker {model.name}")

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=315)
        for tick in ax.get_xticklabels():
            tick.set_rotation(315)
            tick.set_ha('left')
            tick.set_va('top')

        if ylim is not None:
            plt.ylim(*ylim)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.legend()
        plt.grid(axis="y")
        return fig

    def relative_metric_table(self, reference_run_index: int, **kwargs) -> pd.DataFrame:
        pass

    def _metric_table(self, metric_name: List[str],
                                ref: List[MetricReference],
                                mode: List[MetricMode],
                            add_time: bool = False,
                            ) -> pd.DataFrame:
        df = pd.DataFrame(index=[model.name for model in self.models])
        TIME = "time_step"
        for i, metric in enumerate(metric_name):
            metrics = self.get_metrics(metric)
            _ref = ref[i]
            _mode = mode[i]
            for model_name, summary in metrics.items():
                try:
                    if _ref == MetricReference.BEST:
                        x = None
                        if _mode == MetricMode.MIN:
                            x = summary.values['value'].argmin()
                        elif _mode == MetricMode.MAX:
                            x = summary.values['value'].argmax()
                        df.loc[model_name, metric] = summary.values['value'].iloc[x]
                        if add_time:
                            df.loc[model_name, TIME + "_" + metric] = x
                        else:
                            pass
                            #raise NotImplementedError()
                    elif _ref == MetricReference.LAST:
                        df.loc[model_name, metric] = summary.values.iloc[-1]['value']
                        if add_time:
                            df.loc[model_name, TIME + "_" + metric] = summary.values.iloc[-1].name
                    elif _ref == MetricReference.ALL:
                        _metric = [metric + "_" + str(x) for x in summary.values.index]
                        df.loc[model_name, _metric] = summary.values['value'].values
                    else:
                        raise NotImplementedError()#
                except Exception as err:
                    logging.warning(f"Could not compare metric {metric} for model {model_name} due to {err}")
        return df

    def _metric_table_formatting(self, 
                                df: pd.DataFrame, 
                                metric_name: List[str],
                                ref: List[MetricReference],
                                mode: List[MetricMode],
                                add_time: bool = False,
                                max_formatting_decimals: int = 10,
                                best_format: Literal['bold', 'underline'] = 'underline',
                                column_alias: Dict[str, str] = None,
                                custom_formatting: Optional[Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = None,
                                arrow_metric_indicator: bool = True,
                                value_format: Optional[Union[str, List[str]]] = None) -> Styler:
        from awesome.util.format import destinctive_number_float_format
        if value_format is None:
            value_format = []
            # Auto determine
            for column in df.columns:
                col = df[column]
                value_format.append(destinctive_number_float_format(col, max_decimals=max_formatting_decimals))
        else:
            if not isinstance(value_format, list):
                value_format = [value_format for _ in metric_name]
        # Mapping names and aliases

        def _split_number_text(x):
            return x.split(".", maxsplit=1)[1].strip() if "." in x else x

        def _split_number_number(x):
            return int(x.split(".", maxsplit=1)[0])

        names = {x.name: _split_number_text(x.display_name) for x in self.models}
        numbers = {x.name: str(_split_number_number(x.display_name)) + "." for x in self.models}
        ser = pd.Series(numbers)
        ser.name = "number"
        df = pd.concat([df, ser], axis=1)

        if arrow_metric_indicator:
            def _mode_to_arrow(mode: MetricMode):
                if mode == MetricMode.MIN:
                    return '↓'
                elif mode == MetricMode.MAX:
                    return '↑'
                else:
                    return ''
            arrows = {metric_name[i]: _mode_to_arrow(mode[i]) for i in range(len(metric_name))}
            for k, v in arrows.items():
                column_alias[k] = f"{column_alias.get(k, k)} {v}".strip()
        df.index.name = "model"
        df.rename(names, inplace=True, axis=0)
        df.reset_index(inplace=True)
        df.rename(column_alias, inplace=True, axis=1)
        cols = [column_alias.get('number', 'number'), column_alias.get('model', 'model')] + list(df.columns[1: -1])
        df = df[cols]

        best_val = dict()
        for metric, alias in [(k, v) for k, v in column_alias.items() if k in metric_name]:
            val = df[alias].min() if mode[metric_name.index(metric)] == MetricMode.MIN else df[alias].max()
            entries = df[alias] == val
            best_val[alias] = entries

        # Column best
        def _best_mark(x):
            try:
                if x.name == "index":
                    return pd.Series(np.array([''] * len(x)))
                ret = []
                for k, v in x.items():
                    if k in best_val:
                        _v = best_val[k]
                        if _v[x.name]:
                            if best_format == 'underline':
                                ret.append('text-decoration: underline;')
                            elif best_format == 'bold':
                                ret.append('font-weight: bold;')
                            else:
                                raise NotImplementedError()
                        else:
                            ret.append('')
                    else:
                        ret.append('')
                return ret
            except Exception as err:
                print(err)
            return None

        if custom_formatting is not None:
            df = custom_formatting(df)

        formats = {column_alias[metric_name[i]]: value_format[i] for i in range(len(metric_name))}
        style = df.style.apply(_best_mark, axis=1)
        st = style.format(formats, na_rep="-")
        return st


    def metric_table(self,
                     metric_name: Union[str, List],
                     ref: Union[MetricReference, List[MetricReference]] = MetricReference.BEST,
                     mode: Union[Literal['min', 'max'], List] = 'min',
                     add_time: bool = False,
                     formatting: bool = False,
                     max_formatting_decimals: int = 10,
                     best_format: Literal['bold', 'underline'] = 'underline',
                     column_alias: Optional[Dict[str, str]] = None,
                     custom_formatting: Optional[Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = None,
                     arrow_metric_indicator: bool = True,
                     value_format: Optional[Union[str, List[str]]] = None,
                     ) -> pd.DataFrame:
        if column_alias is None:
            column_alias = dict()
        if isinstance(metric_name, str):
            metric_name = [metric_name]
        ref = convert_metric_reference(ref)
        mode = convert_metric_mode(mode)
        if not isinstance(ref, list):
            ref = [ref for _ in metric_name]
        if not isinstance(mode, list):
            mode = [mode for _ in metric_name]

        df = self._metric_table(metric_name=metric_name, ref=ref, mode=mode, add_time=add_time)
        if formatting:
            return self._metric_table_formatting(df=df, 
                                                 metric_name=metric_name, 
                                         ref=ref, 
                                         mode=mode, 
                                         add_time=add_time, 
                                         max_formatting_decimals=max_formatting_decimals, 
                                         best_format=best_format, 
                                         column_alias=column_alias,
                                         custom_formatting=custom_formatting, 
                                         arrow_metric_indicator=arrow_metric_indicator, 
                                         value_format=value_format)
        return df


    def get_metric_df(self, metric_name: str) -> pd.DataFrame:
        """Gets a wide-form dataframe of the metric values,
        whereby each models metrics goes into a column defined by its name.


        Parameters
        ----------
        metric_name : str
            Name of metric to get a df for.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the metric values.
        """
        import pandas as pd
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}
        df = pd.DataFrame(columns=['global_step'] + list(metrics.keys()))

        for model_name, summary in metrics.items():
            frame = summary.values[['value', 'global_step']]
            existing = frame['global_step'].isin(df['global_step'])
            df.loc[df['global_step'].isin(frame['global_step']), model_name] = frame[existing]['value']
            non_existing = frame[~existing]
            non_existing = non_existing.rename(dict(value=model_name), axis=1)
            df = pd.concat([df, non_existing])
        df = df.sort_values("global_step")
        return df

