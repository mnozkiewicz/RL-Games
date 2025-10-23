import numpy as np
import random
from .base_agent import BaseAgent


class DummyAgent(BaseAgent):
    """
    A random player
    """

    def __init__(self, state_space_shape: int, action_space_size: int):
        super().__init__(state_space_shape, action_space_size)

    def get_action(self, state: np.ndarray) -> int:
        assert state.shape[0] == self.state_space_shape, (
            "Missmatch on state space shape"
        )
        return random.randrange(self.action_space_size)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        terminal: bool,
    ): ...
