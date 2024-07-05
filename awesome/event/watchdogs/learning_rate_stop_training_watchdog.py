from typing import Any, Dict, Union
from awesome.event import EventArgs, Watchdog
from awesome.error.stop_training import StopTraining
from torch.optim.optimizer import Optimizer
from awesome.event import TorchModelStepEventArgs
import logging
from enum import Enum


class MODE(Enum):
    """Mode for the learning rate stop training watchdog."""
    LT = "lt"
    """Smaller than."""
    LTE = "lte"
    """Smaller than or equal."""
    GT = "gt"
    """Greater than."""
    GTE = "gte"
    """Greater than or equal."""
    EQ = "eq"
    """Equal."""


class LearningRateStopTrainingWatchdog(Watchdog):
    """Watchdog which stops the training if the learning rate is smaller / greater / equal than the current value in the optimizer."""

    _operation: Dict[MODE, Any] = {
        MODE.LT: lambda x, y: x < y,
        MODE.LTE: lambda x, y: x <= y,
        MODE.GT: lambda x, y: x > y,
        MODE.GTE: lambda x, y: x >= y,
        MODE.EQ: lambda x, y: x == y
    }

    def __init__(self,
                 learning_rate: float = 1e-8,
                 mode: Union[MODE, str] = MODE.LTE,
                 verbose: bool = True):
        """Initializes the watchdog.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate which is used to compare with the current learning rate, by default 1e-8
        mode : Union[Mode, str], optional
            Mode which is used to compare the learning rate, by default MODE.LTE
            Readed as optim.lr [mode] self.learning_rate
        verbose : bool, optional
            If true, the watchdog will print a message if the training is stopped, by default True
        """
        super().__init__()
        if isinstance(mode, str):
            mode = MODE(mode)
        self.learning_rate = learning_rate
        self.mode = mode
        self.verbose = verbose

    def _eval(self, lr: float) -> bool:
        op = self._operation.get(self.mode, None)
        if op is None:
            raise ValueError(f"Mode {self.mode} is not supported!")
        return op(lr, self.learning_rate)

    def __call__(self, ctx: Dict, args: TorchModelStepEventArgs):
        if args.optimizer is not None:
            for group in args.optimizer.param_groups:
                if 'lr' in group:
                    if self._eval(group['lr']):
                        if self.verbose:
                            logging.info(
                                f"Stopping training because learning rate {group['lr']} {self.mode.value} than {self.learning_rate}!")
                        raise StopTraining()
