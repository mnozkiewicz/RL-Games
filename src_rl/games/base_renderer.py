from abc import ABC, abstractmethod
from pygame import Surface
from PIL import Image
from typing import Generic, Optional
from .base_game import GameType


class BaseRenderer(ABC, Generic[GameType]):
    def __init__(self, game: GameType):
        super().__init__()
        self.game = game

    @abstractmethod
    def draw(self, surface: Surface) -> None:
        """
        Draws the current state of self.game to the surface.
        """
        pass

    @abstractmethod
    def game_screenshot(self, output_size: Optional[int]) -> Image.Image:
        """
        Draws the current state of self.game to the surface.
        """
        pass
