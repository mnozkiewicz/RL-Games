from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    An abstract base class (ABC) for a reinforcement learning agent.
    """

    def __init__(self, state_space_shape: int, action_space_size: int):
        """
        Initializes the agent.

        Args:
            action_space_size (int): The number of discrete actions available.
            state_space_shape (Tuple[int, ...]): The shape of the state array.
        """
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.is_eval_mode = False

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        """
        Gets an action for a given state during training.
        This may include exploration (e.g., epsilon-greedy).

        Args:
            state (np.ndarray): The current game state.

        Returns:
            int: The action to take.
        """
        ...

    def get_optimal_action(self, state: np.ndarray) -> int:
        """
        Gets the best possible action for a given state during evaluation.
        This should be deterministic, with no exploration.
        Args:
            state (np.ndarray): The current game state.

        Returns:
            int: The optimal action to take.
        """
        return self.get_action(state)

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
        Args:
            state (np.ndarray): The state before the action.
            action (int): The action taken.
            reward (int): The reward received.
            next_state (np.ndarray): The state after the action.
            done (bool): Whether the episode finished.
        """
        ...

    # def set_evaluation_mode(self, is_eval: bool):

    # def save_model(self, path: str):

    # def load_model(self, path: str):
