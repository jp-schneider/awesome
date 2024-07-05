from dataclasses import dataclass

from awesome.serialization.json_convertible import JsonConvertible

from dataclasses import dataclass, field

@dataclass
class GroundTruthForegroundIdMapping(JsonConvertible):
    """Stores for one dataset / sequence the mapping."""

    dataset_name: str = field()
    """Name of the dataset."""

    trajectory_foreground_id: int = field()
    """Foreground id of the segementation objec of the trajectory files."""

    ground_truth_foreground_id: int = field()
    """Foreground id of the segementation object of the ground truth files."""
