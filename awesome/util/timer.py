import time
from datetime import datetime, timedelta
from awesome.util.format import strfdelta

class Timer:
    """Time class which can be used as context manager to measure times.

    >>> with Timer() as timer:
            ...
        print(timer.duration)

    Will result int the time as timedelta.
    """

    _start: float
    """Start time of timer."""

    _end: float
    """End time of timer"""

    def __init__(self) -> None:
        self._start = -1
        self._end = -1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self) -> float:
        """Starts the timer.
        Can be used in manual mode.

        Returns
        -------
        float
            The time in UTC Timestamp when it was started.
        """
        self._start = time.time()
        return self._start

    def stop(self) -> float:
        """Stops the timer.
        Can be used in manual mode.

        Returns
        -------
        float
            The time in UTC Timestamp when it was stopped.
        """
        self._end = time.time()
        return self._end

    @property
    def start_date(self) -> datetime:
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        return datetime.fromtimestamp(self._start)

    @property
    def end_date(self) -> datetime:
        if self._end == -1:
            raise ValueError(f"Timer was not stopped!")
        return datetime.fromtimestamp(self._end)

    @property
    def duration(self) -> timedelta:
        """The duration between start and stop.
        Raises an error when it was not started / stopped.

        Returns
        -------
        timedelta
            The duration of the measured time.

        Raises
        ------
        ValueError
            If the timer was not started or stopped.
        """
        self._check_values()
        return timedelta(seconds=self._end - self._start)


    @classmethod
    def strfdelta(cls, timedelta: timedelta, format: str) -> str:
        return strfdelta(timedelta, format)

    def elapsed(self) -> timedelta:
        """Returns the elapsed time.
        Can also be accessed when timer is not stopped.

        Returns
        -------
        timedelta
            The elapsed time since start.

        Raises
        ------
        ValueError
            If the timer was not started.
        """
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        return timedelta(seconds=time.time() - self._start)

    def _check_values(self):
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        if self._end == -1:
            raise ValueError(f"Timer was not stopped!")
