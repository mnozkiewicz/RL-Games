from typing import Protocol, runtime_checkable
import pygame


@runtime_checkable  # This decorator lets us use isinstance() at runtime
class IEventHandler(Protocol):
    """
    A protocol (interface) for any object that can handle pygame events.

    An object doesn't need to inherit from this class; it just needs
    to implement the methods defined here.
    """

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Procesess a list of pygame events."""
        ...
