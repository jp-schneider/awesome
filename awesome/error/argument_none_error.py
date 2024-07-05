from typing import Optional
import os


class ArgumentNoneError(AttributeError):
    """Argument None Error for creating an exception with just the argument name and optionally an additional message."""

    def __init__(self, argument_name: str, message: Optional[str] = None) -> None:
        """Creating argument none error with the argument name.

        Parameters
        ----------
        argument_name : str
            The name of the argument which is None.
        message : Optional[str]
            An Additional message wich will be concatenated.
        """
        msg = f"Argument: \"{argument_name}\" is None!"
        if message is not None:
            msg += (os.linesep + message)
        super().__init__(msg)
        self.argument_name = argument_name
        self.additional_message = message
