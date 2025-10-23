from abc import ABC, abstractmethod
from ..games.base_game import BaseGame
import numpy as np


class BasePlayer(ABC):
    def __init__(self, game: BaseGame):
        self.game = game
        self.number_of_moves = self.game.number_of_moves

    @abstractmethod
    def move(self, state: np.ndarray) -> int: ...

    @abstractmethod
    def learn(
        self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool
    ) -> None: ...
