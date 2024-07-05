import base64
import copy
from datetime import datetime
import io
import os
from dataclasses import dataclass, field
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, Type, Union

import jsonpickle
from awesome.measures.tracker_loss import TrackerLoss

from awesome.util.object_factory import ObjectFactory
from awesome.agent.agent import Agent
from awesome.agent.util.tracker import Tracker
from awesome.util.reflection import class_name, dynamic_import
from awesome.serialization import JsonConvertible
from awesome.agent.util.metric_scope import MetricScope
from awesome.error import ArgumentNoneError
from awesome.mixin import FastReprMixin

@dataclass(repr=False)
class BaseAgentCheckpoint(FastReprMixin):

    id: Optional[Any] = field(default=None)
    """Id of the checkpoint when it was restored, otherwise None when from an unsaved agent."""

    name: str = field(default=None)
    """Name of the agent."""

    model_state: Any = field(default=None)
    """A serialized version of the model used, e.g. with torch save or joblib. Handled by subclasses."""

    model_args: Dict[str, Any] = field(default=None)
    """Init arguments with which the model was created. Will be reused to recreate model."""

    model_type: Type = field(default=None)
    """Model class to recreate the model with model args."""

    criterion: Any = field(default=None)
    """The criterion or objective function which is used to train / evalute the model."""

    tracker: Tracker = field(default=None)
    """The model state in terms of progress. Tracks metrics and trained state."""

    dataset_config: Dict[str, Any] = field(default=None)
    """Config arguments for the dataset. These are all arguments which are needed to recreated the dataset."""

    saved_at: Optional[datetime] = None
    """The date where this checkpoint was saved. Default to datetime.now() on object post_init"""

    created_at: Optional[datetime] = None
    """The date where the agent was created."""

    runs_directory: str = field(default=None)
    """The directory where the runs should be stored."""

    agent_directory: Optional[str] = field(default=None)
    """The output folder where the checkpoint was saved."""

    agent_class_name: str = field(default=None)
    """The class name of the agent used for serialization. Is needed to recreate the agent."""

    execution_context: Dict[str, Any] = field(default_factory=dict)
    """Execution context of training containing kwargs to get the model."""

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        return {"model_state", "model_args", "tracker", "criterion", "dataset_config"}

    def __post_init__(self):
        if self.saved_at is None:
            self.saved_at = datetime.now().astimezone()

    def _save_prechecks(self, file_name_or_buf: Union[PathLike[str], BinaryIO]) -> 'BaseAgentCheckpoint':
        """Prechecks which should be done before saving.

        Parameters
        ----------
        file_name_or_buf : Union[PathLike[str], BinaryIO]
            The file name or buffer where the object is saved to.

        Returns
        -------
        BaseAgentCheckpoint
            A copy of the base checkpoint which can be saved.
        """
        # Check for extension
        if isinstance(file_name_or_buf, (PathLike, str)) and "." not in file_name_or_buf:
            # Add default extension
            file_name_or_buf += self.get_default_extension()
        # Create directory
        if isinstance(file_name_or_buf, (PathLike, str)) and not os.path.exists(file_name_or_buf):
            Path(os.path.dirname(file_name_or_buf)).mkdir(parents=True, exist_ok=True)
        # Shallow copy
        
        copied: bool = False
        # Do something to alter if needed
        # If loss has a tracker or logger, remove it
        if isinstance(self.criterion, TrackerLoss):
            ignore = [x for x in self.criterion.__ignore_on_iter__() if not x.startswith("_")]
            for prop in ignore:
                # Set None
                setattr(self.criterion, prop, None)
        else:
            if hasattr(self.criterion, "tracker"):
                self.criterion = copy.copy(self.criterion)
                copied = True
                self.criterion = None
            if hasattr(self.criterion, "logger"):
                if not copied:
                    self.criterion = copy.copy(self.criterion)
                delattr(self.criterion, "logger")
        return self

    def save(self, file_name_or_buf: Union[PathLike[str], BinaryIO]) -> None:
        """Saves the current checkpoint to a dedicated file or stream.

        Parameters
        ----------
        file_name_or_buf : Union[PathLike[str], BinaryIO]
            The file or stream to save to.
        """
        obj = self._save_prechecks(file_name_or_buf)
        serialized = jsonpickle.dumps(obj, keys=True)
        if isinstance(file_name_or_buf, (PathLike, str)):
            with open(file_name_or_buf, "w") as f:
                f.write(serialized)
        else:
            file_name_or_buf.write(serialized.encode())

    def get_default_extension(self) -> str:
        """Gets the file extension which should be used to save the checkpoint

        Returns
        -------
        str
            File extension with .
        """
        return ".jpkl"

    @classmethod
    def load(cls, file_name_or_buf: Union[PathLike[str], BinaryIO]) -> 'BaseAgentCheckpoint':
        """Loads / recreates a serialized version of checkpoint.

        Parameters
        ----------
        file_name_or_buf : Union[PathLike[str], BinaryIO]
            The filename or buffer to load from.

        Returns
        -------
        BaseAgentCheckpoint
            The created base agent checkpoint.
        """
        content = None
        if isinstance(file_name_or_buf, (PathLike, str)):
            with open(file_name_or_buf, "r") as f:
                content = f.read()
        else:
            content = file_name_or_buf.read()
        loaded = jsonpickle.loads(content, keys=True)
        # Create checkpoint
        inst = ObjectFactory.create_from_kwargs(cls, **vars(loaded))
        return inst

    def save_to_string(self) -> str:
        """Saves the checkpoint to a string.

        Returns
        -------
        str
            Serialized checkpoint.
        """
        buf = io.BytesIO()
        self.save(buf)
        buf.seek(0)
        ret = self._encode_buffer(buf.read())
        return ret 
    
    @classmethod
    def load_from_string(cls, s: str) -> 'BaseAgentCheckpoint':
        """Loads the object from a string.

        Parameters
        ----------
        s : str
            The string to load from.

        Returns
        -------
        BaseAgentCheckpoint
            Checkpoint object.
        """
        byt = cls._decode_buffer(s)
        buf = io.BytesIO()
        buf.write(byt)
        buf.seek(0)
        ckp = cls.load(buf)
        return ckp

    def get_additional_checkpoint_information_content(self) -> Dict[str, Any]:
        """Method which can be overriden to get more information which should be placed into the information dict.

        Returns
        -------
        Dict[str, Any]
            Additional information about checkpoint.
        """
        return {}

    def to_agent(self) -> Agent:
        if self.agent_class_name is None:
            raise ArgumentNoneError("agent_class_name")
        cls = dynamic_import(self.agent_class_name)
        if not issubclass(cls, Agent):
            raise ValueError(f"{str(cls)} is not a agent subclass!")
        agent = cls.from_acc(self)
        agent.after_restore(self)
        return agent

    @staticmethod
    def _encode_buffer(buf: bytes) -> str:
        return base64.b64encode(buf).decode()

    @staticmethod
    def _decode_buffer(buf: str) -> bytes:
        return base64.b64decode(buf.encode())
