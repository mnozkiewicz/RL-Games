from abc import ABC, abstractmethod
from typing import Any, Optional
from pygame.event import Event
from pygame import Surface
from PIL import Image
import numpy as np


class BaseGameEngine(ABC):
    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def parse_user_input(self, events: list[Event]) -> list[int]: ...

    @property
    @abstractmethod
    def number_of_moves(self) -> int: ...

    @abstractmethod
    def step(self, actions: list[int]) -> int: ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def draw_state(self, screen: Surface) -> Any: ...

    @abstractmethod
    def game_screenshot(self, output_size: Optional[int]) -> Image.Image: ...

    @abstractmethod
    def processed_state(self) -> np.ndarray: ...
