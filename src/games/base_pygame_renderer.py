from abc import ABC, abstractmethod
from pygame import Surface
from typing import Generic
from .base_game import GameType


class BasePygameRenderer(ABC, Generic[GameType]):
    """
    An abstract base class (ABC) for a game renderer.

    Its main job is to draw the game state onto a pygame `Surface`.
    It can also save the current game state to a picture.
    """

    def __init__(self, game: GameType):
        super().__init__()
        self.game = game

    @abstractmethod
    def init_pygame_renderer(self, surface: Surface) -> None:
        """
        Do any neccesary preprocessing regarding pygame rendering (load assets etc.)
        """
        pass

    @abstractmethod
    def draw(self, surface: Surface) -> None:
        """
        Draws the current state of self.game to the surface.
        """
        pass
