from enum import Enum


class MetricScope(Enum):
    EPOCH = "epoch"
    BATCH = "batch"

    @classmethod
    def display_name(cls, scope: 'MetricScope') -> str:
        """Returns a display name for the mode.

        Parameters
        ----------
        scope : MetricScope
            The scope to get the display name.

        Returns
        -------
        str
            Display name as str.
        """
        if scope == MetricScope.EPOCH:
            return "Epochs"
        if scope == MetricScope.BATCH:
            return "Batches"
        raise NotImplementedError()