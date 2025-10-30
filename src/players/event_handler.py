from typing import Protocol, runtime_checkable
import pygame


@runtime_checkable  # This decorator lets us use isinstance() at runtime
class IEventHandler(Protocol):
    """
    A protocol (interface) for any object that can handle pygame events.
    """

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Procesess a list of pygame events."""
        ...
