from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Set, Optional, Tuple
import jsonpickle
import hashlib
import torch
from datetime import datetime
from awesome.util.path_tools import format_os_independent, relpath
from awesome.serialization.json_convertible import JsonConvertible

@dataclass
class ZooEntry():

    name: str
    """The name of the model or small description."""

    config: Dict[str, Any]
    """Some training configuration of the model."""

    str_repr: str
    """The string representation of the model."""

    query_hash: Optional[str] = field(default=None, repr=False)
    """Hash to find the model in the zoo."""

    state_dict: Optional[Dict[str, Any]] = field(default=None, repr=False)
    """The state dict of the model."""

    context: Optional[Dict[str, Any]] = field(default=None, repr=False)
    """Some context of the model."""

    state_dict_path: Optional[str] = field(default=None, repr=False)
    """Relative path to the state dict of the model. From the config path."""

    created_at: Optional[str] = field(default=datetime.now().astimezone().isoformat(), repr=False)
    """The time when the entry was created."""

    def assemble_state_dict_path(self, config_path: str) -> str:
        """Computes the path to the state dict file.

        Parameters
        ----------
        config_path : str
            The path to the config file.

        Returns
        -------
        str
            The path to the state dict file.
        """
        return os.path.normpath(os.path.join(os.path.dirname(config_path), self.state_dict_path))
        
            
    @staticmethod
    def compute_query_hash(name: str, str_repr:str, config: Dict[str, Any]) -> str:
        """Computes the query hash of the model."""
        # Serialize the config
        config_str = JsonConvertible.convert_to_json_str(config, no_uuid=True, handle_unmatched="jsonpickle")
        hash_object = name + "\n" + str_repr + "\n" + config_str
        # Compute the hash
        return hashlib.sha256(hash_object.encode()).hexdigest()

    @staticmethod
    def compute_config_path(directory: str, query_hash: str) -> str:
        """Computes the path to the config file.

        Parameters
        ----------
        directory : str
            The path to the directory where the zoo entry should be stored.

        query_hash : str
            The query hash of the zoo entry.

        Returns
        -------
        str
            The path to the config file.
        """
        return os.path.join(directory, "config_" + query_hash + ".json")

    def __post_init__(self):
        if self.query_hash is None:
            self.query_hash = ZooEntry.compute_query_hash(self.name, self.str_repr, self.config)
    
    def save(self, directory: str) -> None:
        """Saves the zoo entry to the config and state file.

        Parameters
        ----------
        directory : str
            The path to the directory where the zoo entry should be stored.
        """

        # Compute the path
        config_path = ZooEntry.compute_config_path(directory, self.query_hash)

        # Compute the state dict path if needed
        if self.state_dict_path is None:
            st_path = format_os_independent(os.path.join(directory, "state_dict_" + self.query_hash + ".pth"))
            self.state_dict_path = format_os_independent(relpath(config_path, st_path, is_from_file=True, is_to_file=True))
            
        config_dict = dict(vars(self))
        config_dict.pop("state_dict")
        config_dict.pop("context")

        # Save the config
        with open(config_path, "w") as f:
            f.write(jsonpickle.encode(config_dict, indent=4))

        self.store_state_dict(config_path)
        # Remove reference to free memory
        self.state_dict = None 
        self.context = None 
        

    def load_state_dict(self, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads the state dict of the zoo entry from the state dict path.

        Parameters
        ----------
        config_path : str
            The path to the config file.

        Returns
        ----------
        Tuple[Dict[str, Any], Dict[str, Any]]
            1. The state dict which was loaded.
            2. The context which was loaded.
        """
        save_dict = torch.load(self.assemble_state_dict_path(config_path), map_location=torch.device('cpu'))
        state_dict = save_dict["state_dict"]
        context = save_dict["context"]
        return state_dict, context

    def store_state_dict(self, config_path: str) -> None:
        """Stores the state dict and the context of the zoo entry to the state dict path.
        """
        torch.save(dict(state_dict=self.state_dict, context=self.context), self.assemble_state_dict_path(config_path))

    def delete_state_dict(self) -> None:
        """Deletes the state dict of the zoo entry from the state dict path.
        """
        os.remove(self.state_dict_path)

    def delete(self, directory: str) -> None:
        """Deletes the zoo entry from the config and state file."""
        # Compute the path
        config_path = os.path.join(directory, "config_" + self.query_hash + ".json")

        # Compute the state dict path if needed
        if self.state_dict_path is None:
            self.state_dict_path = os.path.join(directory, "state_dict_" + self.query_hash + ".pth")
    
        # Remove the files if they exist
        if os.path.exists(config_path):
            os.remove(config_path)
        if os.path.exists(self.state_dict_path):
            os.remove(self.state_dict_path)
        
    @staticmethod
    def load(path: str, load_state_dict: bool = False) -> "ZooEntry":
        """Loads the zoo entry from the config and state file.

        Parameters
        ----------
        path : str
            The path to the file where the zoo entry should be loaded from.

        Returns
        -------
        ZooEntry
            The loaded zoo entry.
        """

        # Load the config
        with open(path, "r") as f:
            config_dict = jsonpickle.decode(f.read())

        entry = ZooEntry(**config_dict)

        if load_state_dict:
            entry.state_dict, entry.context = entry.load_state_dict(path)

        return entry

class Zoo():
    """Simple folder based model zoo which stores checkpoints and training configuration of models."""

    def __init__(self, zoo_folder: str = "./data/zoo", **kwargs):
        """Initializes the zoo.

        Parameters
        ----------
        zoo_folder : str, optional
            The folder where the zoo should be stored, by default "./zoo"
        """
        self.zoo_folder = zoo_folder
        # Create the zoo folder if it does not exist
        if not os.path.exists(self.zoo_folder):
            os.makedirs(self.zoo_folder)
        self.files = self.scan()


    def __ignore_on_iter__(self) -> Set[str]:
        return {"files"}

    def scan(self, **kwargs) -> Set[str]:
        """Scans the zoo for all available models.

        Returns
        -------
        Set[str]
            A set of all files within the zoo folder.
        """
        return set([f for f in os.listdir(self.zoo_folder) if os.path.isfile(os.path.join(self.zoo_folder, f))])

    def query_state(self, name: str, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Queries the zoo for a specific model state dict.

        Parameters
        ----------
        name : str
            The name of the model.

        model : torch.nn.Module
            The model which should be queried.

        config : Optional[Dict[str, Any]], optional
            The config of the model, by default None

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
            1. The state dict of the model if it exists, otherwise None.
            2. The context of the model if it exists, otherwise None.
        """
        if config is None:
            config = dict()
        # Compute the query hash
        query_hash = ZooEntry.compute_query_hash(name, repr(model), config)
        path = ZooEntry.compute_config_path(self.zoo_folder, query_hash)
       
        # Check if the query hash exists
        if path in self.files:
            try:
                entry = ZooEntry.load(path, load_state_dict=True)
                return entry.state_dict, entry.context
            except FileNotFoundError as err:
                # File was not found, remove it from the zoo
                # Display a warning
                logging.warning(f"File {path} was not found. Removing it from the zoo.")
                self.files = self.scan()
                return None, None
        else:
            return None, None
        
    def load_model_state(self, name: str, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Loads the model state dict from the zoo if found.

        Parameters
        ----------
        name : str
            The name of the model.

        model : torch.nn.Module
            The model which should be queried.

        config : Optional[Dict[str, Any]], optional
            The (training) config of the model, by default None

        Returns
        -------
        bool
            True if the model was found and loaded, otherwise False.
        """
        state_dict, context = self.query_state(name, model, config)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            return True, context
        else:
            return False, None
        
    def save_model_state(self, name: str, 
                          model: torch.nn.Module, 
                          config: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None
                          ) -> None:
        """Stores the model state dict to the zoo.

        Parameters
        ----------
        name : str
            The name of the model.

        model : torch.nn.Module
            The model which should be stored.

        config : Optional[Dict[str, Any]], optional
            The (training) config of the model, by default None
        """
        if config is None:
            config = dict()
        if context is None:
            context = dict()
        # Compute the query hash
        query_hash = ZooEntry.compute_query_hash(name, repr(model), config)
        # Create the zoo entry
        entry = ZooEntry(name=name, 
                         config=config, 
                         str_repr=repr(model), 
                         query_hash=query_hash, 
                         state_dict=model.state_dict(), 
                         context = context)
        # Save the zoo entry
        entry.save(self.zoo_folder)
        # Update the files
        self.files.add(ZooEntry.compute_config_path(self.zoo_folder, query_hash))