
from enum import Enum
from awesome.agent.util.metric_scope import MetricScope


class LearningScope(Enum):
    """Learning scope of the agent."""

    EPOCH = 'epoch'

    BATCH = 'batch'

    @staticmethod
    def to_metric_scope(mode: 'LearningScope') -> MetricScope:
        if mode == LearningScope.EPOCH:
            return MetricScope.EPOCH
        if mode == LearningScope.BATCH:
            return MetricScope.BATCH
        raise ValueError(f"Unsupported matching of LearningScope {mode} to a metric scope.")
