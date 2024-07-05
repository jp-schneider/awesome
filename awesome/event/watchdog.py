from typing import Any, Optional, Dict
from awesome.event import Event, EventArgs
from abc import abstractmethod


class Watchdog:
    """A Watchdog does some action ("Wuff"!) when it gets poked. 
    Used in combination with the event system.
    Its just a simple wrapper around the event system, which can give eventhandlers more context.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._events = []

    @abstractmethod
    def __call__(self, ctx: Dict[str, Any], args: EventArgs):
        pass

    def register(self, *events: Event):
        """Registers the Watchdog to one or multiple events.
        """
        for event in events:
            if event is None:
                return
            event.attach(self.__call__)
            self._events.append(event)
    
    def remove(self, *events: Event):
        """Removes the watchdog from one or multiple events."""
        for event in events:
            if event in list(self._events):
                event.remove(self.__call__)
                self._events.remove(event)