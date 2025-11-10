from abc import ABC, abstractmethod
from ..games.base_game import BaseGame
import numpy as np


class BasePlayer(ABC):
    """
    An abstract class for any player (human or AI).
    This interface defines the methods needed to interact with game engine.
    """

    def __init__(self, game: BaseGame):
        super().__init__()
        self.game = game
        self.number_of_moves = self.game.number_of_moves

    @abstractmethod
    def move(self, state: np.ndarray) -> int:
        """
        Determines the next action based on the current game state.
        """
        ...

    @abstractmethod
    def feedback(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        """
        Provides the player with the results of its last action.

        This method enables learning during visualisation.
        A human player might simply ignore it.
        """
        ...
