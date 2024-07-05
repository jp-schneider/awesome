from typing import Any, TYPE_CHECKING, Optional
import torch
from torch.utils.data.dataset import Dataset
from abc import abstractmethod

if TYPE_CHECKING:
    from awesome.agent.torch_agent import TorchAgent
else:
    TorchAgent = Any  # Avoiding import error


class PretrainableModule():
    """Marking that the module can be pre-trained."""

    @abstractmethod
    def pretrain(self,
                 train_set: Dataset,
                 test_set: Dataset,
                 device: torch.device,
                 agent: TorchAgent,
                 use_progress_bar: bool = True,
                 **kwargs
                 ) -> Any:
        """Method to pretrain the module.

        Parameters
        ----------
        train_set : Dataset
            The training dataset
        test_set : Dataset
            The test dataset
        device : torch.device
            The device of the module
        agent : TorchAgent
            The agent which invoked the pretraining
        use_progress_bar : bool, optional
            If a progressbar should be used, by default True
        **kwargs
            Additional parameters       
        Returns
        -------
        Any
            The state for the model to use for pretraining.
        """

    @abstractmethod
    def pretrain_load_state(self,
                            train_set: Dataset,
                            test_set: Dataset,
                            device: torch.device,
                            agent: TorchAgent,
                            state: Any,
                            use_progress_bar: bool = True,
                            do_pretrain_checkpoints: bool = False,
                            pretrain_checkpoint_dir: Optional[str] = None, 
                            **kwargs):
        """Load the state for pretraining.
        Gets called with the state returned by pretrain.

        Parameters
        ----------
        train_set : Dataset
            The training dataset
        test_set : Dataset
            The test dataset
        device : torch.device
            The device of the module
        agent : TorchAgent
            The agent which invoked the pretraining
        
        state : Any
            The state returned by pretrain which should be applied to the module to
            get the pretraining state.
        use_progress_bar : bool, optional
            If a progressbar should be used, by default True
        do_pretrain_checkpoints : bool, optional
            If pretrain checkpoints should be done, by default False
        pretrain_checkpoint_dir : Optional[str], optional
            The directory to save the pretrain checkpoints to, by default None
        **kwargs
            Additional parameters
        """
        pass
