from typing import Dict, Any
from abc import ABC


class BatchedDataset(ABC):
    """Dataset that supports splitting in batches and defining indices."""

    def __init__(self, batch_size: int = 32, shuffle_in_dataloader: bool = True, **kwargs) -> None:
        super(BatchedDataset, self).__init__(**kwargs)
        self.training_batch_size = kwargs.get('training_batch_size') if kwargs.get(
            'training_batch_size', None) else batch_size
        self.validation_batch_size = kwargs.get('validation_batch_size') if kwargs.get(
            'validation_batch_size', None) else batch_size
        self.shuffle_in_training_dataloader = kwargs.get('shuffle_in_training_dataloader') if kwargs.get(
            'shuffle_in_training_dataloader', None) else shuffle_in_dataloader
        self.shuffle_in_validation_dataloader = kwargs.get('shuffle_in_validation_dataloader') if kwargs.get(
            'shuffle_in_validation_dataloader', None) else shuffle_in_dataloader

    @property
    def batch_size(self):
        """Overall batch size."""
        return self.training_batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.training_batch_size = value
        self.validation_batch_size = value

    @property
    def shuffle_in_dataloader(self):
        """If the data should be shuffled in the dataloader."""
        return self.shuffle_in_training_dataloader

    @shuffle_in_dataloader.setter
    def shuffle_in_dataloader(self, value: bool):
        self.shuffle_in_training_dataloader = value
        self.shuffle_in_validation_dataloader = value