

from typing import Any, Dict, Optional, Set

from awesome.agent.util.tracker import Tracker

class _NOTSET():
    pass

NOTSET = _NOTSET()

class TrackerLoss():
    """The tracker loss is a loss class for debugging purposes allowing to make fine grained loss tracking possible."""

    _tracker: Tracker

    _logger: Any

    _is_epoch: bool

    _is_training: bool

    _call_history: Dict[str, Dict[int, int]]

    _name_path: str

    _origin_name: str

    def __init__(self, 
                 tracker: Optional[Any] = None,
                 logger: Optional[Any] = None, 
                 is_epoch: bool = False,
                 is_training: bool = True,
                 **kwargs) -> None:
        self._logger = logger
        self._tracker = tracker
        self._is_epoch = is_epoch
        self._is_training = is_training
        self._call_history = dict()
        self._name_path = ""
        self._origin_name = None

    def __ignore_on_iter__(self) -> Set[str]:
        ret = set()
        ret.add("logger")
        ret.add("tracker")
        ret.add("_tracker")
        ret.add("_logger")
        ret.add("_is_epoch")
        ret.add("is_epoch")
        ret.add("is_training")
        ret.add("_is_training")
        ret.add("_call_history")
        ret.add("call_history")
        ret.add("_name_path")
        ret.add("name_path")
        ret.add("_origin_name")
        ret.add("origin_name")
        return ret

    def log(self, name: str, value: Any, **kwargs) -> None:
        """Logs a value to tensorboard and tracker.

        This is a method for debugging purposes allowing to make fine grained loss tracking possible.

        Parameters
        ----------
        
        name : str
            The name of the value.

        value : Any
            The value to log.
        """
        if self.tracker is None:
            return
        if self.logger is None:
            return
       
        step = self.tracker.global_steps if self.is_training else self.tracker.global_epochs
        tag = self._get_tag(name, step)
        self.logger.log_value(tag=tag, value=value, step=step)

    def _get_tag(self, name: str, step:int) -> str:
        path_name = self._name_path if self._name_path is not None else ""
        suffix_name = path_name + ("." if len(path_name) > 0 else "") + name
        metric_name = self.origin_name
        tracker_tag = Tracker.assemble_tag(metric_name, 
                                   in_training=self.is_training,
                                   is_epoch=self.is_epoch).split("/")[::-1]
        
        tag = "/".join(tracker_tag) + "/" + suffix_name

        suffix = ""
        if self._call_history is None:
            self._call_history = dict()
        if tag in self._call_history:
            val = self._call_history[tag]
            if step in val:
                val[step] += 1
                suffix = f"_{val[step]}"
            else:
                # Reset all counters for this tag
                val.clear()
                val[step] = 0
        else:
            self._call_history[tag] = {step: 0}
        return tag + suffix


    @property
    def tracker(self) -> Tracker:
        return self._tracker
    
    @tracker.setter
    def tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker
        self._recursive_set("_tracker", tracker)

    @property
    def logger(self) -> Any:
        return self._logger
    
    @logger.setter
    def logger(self, logger: Any) -> None:
        self._logger = logger
        self._recursive_set("_logger", logger)

    @property
    def is_epoch(self) -> bool:
        return self._is_epoch
    
    @is_epoch.setter
    def is_epoch(self, is_epoch: bool) -> None:
        self._is_epoch = is_epoch
        self._recursive_set("_is_epoch", is_epoch)

    @property
    def is_training(self) -> bool:
        return self._is_training
    
    @is_training.setter
    def is_training(self, is_training: bool) -> None:
        self._is_training = is_training
        self._recursive_set("_is_training", is_training)

    @property
    def call_history(self) -> Dict[str, Dict[int, int]]:
        return self._call_history
    
    @call_history.setter
    def call_history(self, call_history: Dict[str, Dict[int, int]]) -> None:
        self._call_history = call_history
        self._recursive_set("_call_history", call_history)

    @property
    def name_path(self) -> str:
        return self._name_path
    
    @name_path.setter
    def name_path(self, name_path: str) -> None:
        self._name_path = name_path
        self._recursive_set_name_path(name_path)

    @property
    def origin_name(self) -> str:
        if self._origin_name is None:
            return self.get_name()
        return self._origin_name

    @origin_name.setter
    def origin_name(self, origin_name: str) -> None:
        self._origin_name = origin_name
        self._recursive_set("_origin_name", origin_name)

    def _recursive_set(self, 
                       name: property, 
                       value: Any, 
                       memo: Set[Any] = None):
        if memo is None:
            memo = set()
        if self in memo:
            return
        else:
            memo.add(self)
        # Traverse children
        for key, child in self.__dict__.items():
            if child is None:
                continue
            if isinstance(child, TrackerLoss):
                setattr(child, name, value)
                child._recursive_set(name, value, memo)

    def _recursive_set_name_path(self, 
                       prefix: Optional[str] = NOTSET,
                       origin_name: Optional[str] = NOTSET, 
                       memo: Set[Any] = None):
        if prefix == NOTSET:
            prefix = ""
        if origin_name == NOTSET:
            origin_name = self.origin_name
        if memo is None:
            memo = set()
        if self in memo:
            return
        else:
            memo.add(self)
        # Traverse children
        for prop, child in self.__dict__.items():
            if child is None:
                continue
            if isinstance(child, TrackerLoss):
                child_prefix = None
                if prefix is not None:
                    child_prefix = prefix + ("." if len(prefix) > 0 else "") + prop 
                child._name_path = child_prefix
                child._origin_name = origin_name
                child._recursive_set_name_path(prefix=child_prefix, origin_name=origin_name, memo=memo)

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return type(self).__name__
