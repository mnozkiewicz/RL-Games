import random
import numpy as np

from src_rl.games.base_game import BaseGame
from .base_player import BasePlayer


class AIPlayer(BasePlayer):
    def __init__(self, game: BaseGame):
        super().__init__(game)

    def move(self, state: np.ndarray) -> int:
        return random.randrange(self.number_of_moves)

    def learn(
        self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool
    ) -> None:
        pass
