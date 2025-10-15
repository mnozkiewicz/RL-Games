from abc import ABC, abstractmethod
from typing import Any
from pygame.event import Event
from pygame import Surface


class BaseGameEngine(ABC):
    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def parse_user_input(self, events: list[Event]) -> list[int]: ...

    @abstractmethod
    def step(self, actions: list[int]): ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def draw_state(self, screen: Surface) -> Any: ...

    @abstractmethod
    def get_state(self) -> Any: ...
