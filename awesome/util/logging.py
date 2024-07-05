import logging
import sys
from awesome.util.package_tools import get_package_name

logger: logging.Logger = logging.getLogger(get_package_name())
"""Package logger for the current package."""

def basic_config(level: int = logging.INFO):
    """Basic logging configuration with sysout logger.

    Parameters
    ----------
    level : logging._Level, optional
        The logging level to consider, by default logging.INFO
    """
    root = logging.getLogger()
    root.setLevel(level)
    _fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    _date_fmt = '%Y-%m-%d:%H:%M:%S'
    logging.basicConfig(format=_fmt,
                        datefmt=_date_fmt,
                        level=level)
    fmt = logging.Formatter(_fmt, _date_fmt)
    root.handlers[0].setFormatter(fmt)
    # Set default for other loggers
    if "matplotlib" in sys.modules:
        logger = logging.getLogger("matplotlib")
        logger.setLevel(logging.WARNING)

