import logging
from typing import List, Tuple
import random
from abc import ABC
from sklearn.model_selection import train_test_split
import numpy as np
import os
from awesome.serialization.json_convertible import JsonConvertible

class SeparableDataset(ABC):

    def __init__(self, 
                 split_seed: int = 946817, 
                 split_ratio: float = 0.7, 
                 indices_file: str = None,
                 **kwargs) -> None:
        """Splittable dataset which implements a seed and split ration.

        Parameters
        ----------
        split_seed : int, optional
            Seed to split data with, by default 946817
        split_ratio : float, optional
            The ratio of train data, the test data will be then 1 - split_ratio., by default 0.7
        indices_file : str, optional
            File to store the splitted indices in, by default None
            Should be a yaml or json file.
        """
        super(SeparableDataset, self).__init__(**kwargs)
        self.split_seed = split_seed
        self.split_ratio = split_ratio
        self.indices_file = indices_file
        self.training_indices = None
        if "training_indices" in kwargs:
            self.training_indices = kwargs.get("training_indices")
        self.validation_indices = None
        if "validation_indices" in kwargs:
            self.validation_indices = kwargs.get("validation_indices")

    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the split indices when splitting a dataset in its train and test set.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array with train and test indices. 
        """
        if self.training_indices is None or self.validation_indices is None:
            logging.debug(f"Split indices with seed {self.split_seed} and ratio {self.split_ratio}.")
            self.training_indices, self.validation_indices = self._split_indices()
        return self.training_indices, self.validation_indices

    def _split_indices(self) -> Tuple[List[int], List[int]]:
        # If indices file is given, load indices from file if exists, else generate new indices
        
        loaded_indices = False
        train_idx = None
        test_idx = None

        if self.indices_file is not None:
            if os.path.exists(self.indices_file):
                index_obj = JsonConvertible.load_from_file(self.indices_file)
                train_idx = np.array(index_obj['training_indices']).astype(int)
                test_idx = np.array(index_obj['validation_indices']).astype(int)
                train_idx = np.sort(train_idx)
                test_idx = np.sort(test_idx)
                loaded_indices = True
                logging.info(f"Loaded splitted indices from {self.indices_file}.")
            else:
                pass
       
        if not loaded_indices:
            if not 0.0 <= self.split_ratio <= 1.0:
                raise ValueError(
                    f"Split ratio {self.split_ratio} is invalid, must be in the interval [0., 1.]")
            if self.split_seed is not None:
                random.seed(self.split_seed)

            n_train = int(self.split_ratio * len(self))
            if n_train == 0:
                test_idx = np.arange(0, len(self), 1, dtype=int)
                train_idx = np.arange(0, 0, 1, dtype=int)
            n_test = len(self) - n_train
            if n_test == 0:
                train_idx = np.arange(0, len(self), 1, dtype=int)
                test_idx = np.arange(0, 0, 1, dtype=int)
            if n_train != 0 and n_test != 0:
                train_idx, test_idx = train_test_split(np.arange(0, len(self),
                                                                1, dtype=int),
                                                    train_size=n_train, test_size=n_test,
                                                    random_state=self.split_seed, shuffle=True)
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)

        
        # Store indices in file
        if self.indices_file is not None and not loaded_indices:
            index_obj = {
                "training_indices": train_idx.tolist(),
                "validation_indices": test_idx.tolist()
            }
            JsonConvertible.convert_to_file(index_obj, self.indices_file)
            logging.info(f"Saved splitted indices to {self.indices_file}.")

        return train_idx, test_idx
