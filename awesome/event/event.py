from typing import Any, Callable, Dict, Optional, TypeVar, Generic, List
from awesome.event.event_args import EventArgs
import logging

T = TypeVar('T', bound=EventArgs)


class Event(Generic[T]):
    """Event class mimiking in the observer pattern.
    """

    def __init__(self, source: Any, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.context: Dict[str, Any] = {}
        """Context of the event which will be provided along the arguments. By default there is a source key defining the event emmitter."""
        if context is not None:
            self.context = context
        if self.context.get('source', None) is None:
            self.context['source'] = source
        self._observers: List[Callable[[Dict[str, Any], T], None]] = []
        """Observers (Handlers) which should be notified, called on event notification."""
        self.muted = False
        """If the event is muted, so it will not be invoked."""

    def attach(self, function: Callable[[Dict[str, Any], T], None]):
        """Attaches a function (handle) to the event. It should receive 2 arguments, the first will be context
        and the second the event arguments defined by the current event.

        Example:
        >>> def _my_handle(ctx: Dict[str, Any], args: MyEventArgs):
                ...

        Parameters
        ----------
        function : Callable[[Dict[str, Any], T], None]
            The function / handle to invoke.

        Raises
        ------
        ValueError
            If function is not callable.
        """
        if not isinstance(function, Callable):
            raise ValueError(
                f"Function has to wrong type! Expected: Callable[[Dict[str, Any], T], None]")
        self._observers.append(function)

    @property
    def observers(self) -> List[Callable[[Dict[str, Any], T], None]]:
        """Returning a list of observers.

        Returns
        -------
        List[Callable[[Dict[str, Any], T], None]]
            The list of observers for the current event.
        """
        return list(self._observers)

    def has_listener(self) -> bool:
        """Get whether the event has listerners.

        Returns
        -------
        bool
            If there are listerners
        """
        return len(self._observers) > 0

    def remove(self, function: Callable[[Dict[str, Any], T], None]):
        """Removing the given function from the event.

        Parameters
        ----------
        function : Callable[[Dict[str, Any], T], None]
            The function to remove.
        """
        self._observers.remove(function)

    def notify(self, args: T, **kwargs):
        """Notifies all observers with the given args and context.

        Parameters
        ----------
        args : T
            The arguments to invoke.
        kwargs 
            Other kwargs which will be also part of the context.
        """
        if args.cancel:
            # Kinda stupid, but could happen....
            return
        if self.muted:
            return
        ctx = dict(self.context)
        ctx.update(kwargs)
        for i, fnc in enumerate(self._observers):
            if args.cancel:
                logging.debug("Event with args: %s was canceled by observer %s with name %s" % (
                    str(args), str(i-1), self._observers[i-1]))
                break
            fnc(ctx, args)
