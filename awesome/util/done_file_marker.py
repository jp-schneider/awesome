import os
from typing import List, Optional, Type, Tuple
from awesome.util.logging import logger
from datetime import datetime


class DoneFileMarker():
    """Context manager to create a file that indicates that a process is done."""

    def __init__(self,
                 directory: str,
                 identifier: str):
        self.directory = directory
        self.identifier = identifier
        self.path = DoneFileMarker.get_marker_file(identifier, directory)

    @classmethod
    def get_marker_file(cls, identifier: str, directory: str) -> str:
        return os.path.join(directory, f".{identifier}.done")

    def mark(self, message: str):
        with open(self.path, "w") as f:
            f.write(message)

    def read(self) -> str:
        with open(self.path, "r") as f:
            return f.read()

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def delete(self):
        if self.exists():
            os.remove(self.path)

    def notify_if_exists(self, suffix: Optional[str] = None) -> bool:
        if self.exists():
            logger.info(f"Markerfile: {self.identifier} found." +
                        (f" {suffix}" if suffix is not None else ""))
            return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If there is no exception, save a done file
        if exc_type is None:
            self.mark(
                f"This file indicates that process for {self.identifier} is completed. \nDate {datetime.now().strftime('%y-%m-%d %H:%M:%S')}.\n")
        else:
            if self.exists():
                self.delete()

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.__str__()
