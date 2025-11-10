import numpy as np

from .base_player import BasePlayer
from ..games.base_game import BaseGame
from ..agents.base_agent import BaseAgent


class AIPlayer(BasePlayer):
    """
    A 'Player' that is controlled by an 'Agent'.

    This class acts as an adapter, bridging the `BasePlayer` and BaseAgent`
    interfaces
    """

    def __init__(self, game: BaseGame, agent: BaseAgent, learn: bool = False) -> None:
        """
        If 'learn' is set to True, then the agent will learn during visualisation
        """
        super().__init__(game)
        self.agent = agent
        self.learn = learn

        if self.learn:
            self.agent.set_train_mode()
        else:
            self.agent.set_eval_mode()

    def move(self, state: np.ndarray) -> int:
        return self.agent.choose_action(state)

    def feedback(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        if self.learn:
            self.agent.learn(state, action, reward, next_state, terminal)
