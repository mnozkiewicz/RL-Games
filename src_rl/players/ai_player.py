import numpy as np

from src_rl.games.base_game import BaseGame
from .base_player import BasePlayer
from ..agents.base_agent import BaseAgent


class AIPlayer(BasePlayer):
    def __init__(self, game: BaseGame, agent: BaseAgent):
        super().__init__(game)
        self.agent = agent

    def move(self, state: np.ndarray) -> int:
        return self.agent.choose_action(state)
