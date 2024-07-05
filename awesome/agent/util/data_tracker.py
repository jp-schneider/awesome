from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class DataTracker():
    """The data tracker is used to store the data from a data loader in a combined manner to tighten training logic.
    Usally it preserves the data accross an epoch for logging."""

    track_loss: bool = field(default=True)
    """If the tracker should track the loss at all."""

    track_prediction: bool = field(default=False)
    """If the tracker should track the prediction at all."""

    track_label: bool = field(default=False)
    """If the tracker should track the label at all."""

    track_input: bool = field(default=False)
    """If the tracker should track the input at all."""

    track_indices: bool = field(default=False)
    """If the tracker should track the indices at all."""

    track_statistics: bool = field(default=True)
    """If the tracker should track the statistics at all."""

    loss: List[torch.Tensor] = field(default_factory=list, init=False)
    """List containing the loss of the training run."""

    prediction: List[torch.Tensor] = field(default_factory=list, init=False)
    """List containing the predictions of a training epoch."""

    label: List[torch.Tensor] = field(default_factory=list, init=False)
    """List containing the labels of a training epoch."""

    input: List[torch.Tensor] = field(default_factory=list, init=False)
    """List containing the input of a training epoch."""

    indices: List[torch.Tensor] = field(default_factory=list, init=False)
    """List containing the indices of a training epoch."""

    def push(self, loss: Optional[torch.Tensor] = None,
             prediction: Optional[torch.Tensor] = None,
             label: Optional[torch.Tensor] = None,
             input: Optional[torch.Tensor] = None,
             indices: Optional[torch.Tensor] = None,
             ):
        """Adds data to the tracker and considering whether it should track.

        Parameters
        ----------
        loss : Optional[torch.Tensor], optional
            The loss to track, by default None
        prediction : Optional[torch.Tensor], optional
            The prediction to track, by default None
        label : Optional[torch.Tensor], optional
            The label to track, by default None
        input : Optional[torch.Tensor], optional
            The input to track, by default None
        indices : Optional[torch.Tensor], optional
            The indices to track, by default None
        statistics : Optional[StatisticTensor], optional
            The statistics to track, by default None
        """
        if loss is not None and self.track_loss:
            self.loss.append(loss)
        if prediction is not None and self.track_prediction:
            self.prediction.append(prediction)
        if label is not None and self.track_label:
            self.label.append(label)
        if input is not None and self.track_input:
            self.input.append(input)
        if indices is not None and self.track_indices:
            self.indices.append(indices)

    @property
    def running_loss(self) -> float:
        """Computes the mean loss."""
        if len(self.loss) > 0:
            return sum(self.loss) / len(self.loss)
        else:
            return 0.0

    def combined_predictions(self) -> torch.Tensor:
        if not self.track_prediction:
            return None
        return torch.cat(self.prediction)

    def combined_labels(self) -> torch.Tensor:
        if not self.track_label:
            return None
        return torch.cat(self.label)

    def combined_inputs(self) -> torch.Tensor:
        if not self.track_input:
            return None
        return torch.cat(self.input)

    def combined_indices(self) -> torch.Tensor:
        if not self.track_indices:
            return None
        return torch.cat(self.input)
