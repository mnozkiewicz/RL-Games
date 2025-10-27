from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar


class BaseGame(ABC):
    """
    An abstract base class (ABC) defining the interface for all game environments.

    All games used in this reinforcement learning project must inherit from
    this class and implement its abstract methods. This ensures compatibility
    with the RL engine, renderers, and other components.
    """

    @abstractmethod
    def reset(self):
        """
        Resets the game to its initial state.

        This method is called at the beginning of each new episode.
        The game's state should be queryable via `processed_state()`
        after calling this.
        """
        ...

    @property
    @abstractmethod
    def number_of_moves(self) -> int:
        """
        Gets the total number of possible distinct actions in the game.

        Returns:
            int: The size of the action space (e.g., 4 for Up, Down, Left, Right).
        """
        ...

    @abstractmethod
    def step(self, action_label: int) -> int:
        """
        Performs one time step of the game logic given a list of actions.

        Args:
            actions (list[int]): A list of actions to be taken in this step.
                The interpretation of this list is game-specific
                (e.g., one action per player, or a set of simultaneous actions).

        Returns:
            int: The reward (or score change) obtained from performing the action(s).
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """
        Checks if the game is still in progress.

        Returns:
            bool: True if the game is running (episode is not finished),
                  False if the game is over (e.g., player died, game won).
        """
        ...

    @abstractmethod
    def processed_state(self) -> np.ndarray:
        """
        Gets the current state of the game, processed for the neural network.

        Returns:
            np.ndarray: A numerical representation of the game state,
                suitable for input into a model.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """
        Returns the game's name to be displayed
        Returns:
            str: name of the game
        """
        ...


GameType = TypeVar("GameType", bound=BaseGame)
