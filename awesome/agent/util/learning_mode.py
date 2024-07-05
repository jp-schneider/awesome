from enum import Enum
from awesome.agent.util.metric_mode import MetricMode


class LearningMode(Enum):
    """Defining the learning mode for a agent."""

    INFERENCE = "inference"

    TRAINING = "training"

    VALIDATION = "validation"

    @staticmethod
    def to_metric_mode(mode: 'LearningMode') -> MetricMode:
        if mode == LearningMode.TRAINING:
            return MetricMode.TRAINING
        if mode == LearningMode.VALIDATION:
            return MetricMode.VALIDATION
        raise ValueError(f"Unsupported matching of LearningMode {mode} to a metric mode.")
