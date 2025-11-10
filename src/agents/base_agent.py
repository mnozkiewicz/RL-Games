from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    An abstract base class (ABC) for a reinforcement learning agent.
    """

    def __init__(self) -> None:
        """
        Initializes the agent.
        The agent is by default set to training mode.
        """
        super().__init__()
        self.eval_mode = False

    @abstractmethod
    def choose_action(self, state: np.ndarray) -> int:
        """
        Gets an action for a given state. Function used during training,
        but also during evaluation, if self.eval_mode is set tu True
        """
        ...

    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        """
        Performs a learning step based on a transition.
        A transition is a typical SARSA tuple (state, action, reward, next_state),
        and also information wheter the ended the game.
        """
        ...

    @abstractmethod
    def set_eval_mode(self) -> None: ...

    @abstractmethod
    def set_train_mode(self) -> None: ...

    @abstractmethod
    def save_model(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device: str = "cpu") -> BaseAgent: ...
