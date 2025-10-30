from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    An abstract base class (ABC) for a reinforcement learning agent.
    """

    def __init__(self, state_space_shape: int, action_space_size: int):
        """
        Initializes the agent.
        The agent is by default set to training mode.
        """
        super().__init__()
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
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
        reward: int,
        next_state: np.ndarray,
        terminal: bool,
    ):
        """
        Performs a learning step based on a transition.
        A transition is a typical SARSA tuple (state, action, reward, next_state),
        and also information wheter the ended the game.
        """
        ...

    @abstractmethod
    def set_eval_mode(self): ...

    @abstractmethod
    def set_train_mode(self): ...

    @abstractmethod
    def save_model(self, path: str): ...

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device: str = "cpu") -> "BaseAgent": ...
