from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Literal


class BaseGame(ABC):
    """
    An abstract base class (ABC) defining the interface for all game environments.

    All games used in this reinforcement learning project must inherit from
    this class and implement its abstract methods.
    """

    def __init__(
        self,
        state_type: Literal["processed_state", "raw_pixels"],
        infinite: bool = False,
        is_ai_controlled: bool = False,
    ) -> None:
        super().__init__()
        self._use_processed_state = state_type == "processed_state"

    @abstractmethod
    def reset(self) -> None:
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
        """
        ...

    @abstractmethod
    def step(self, action_label: int) -> float:
        """
        Performs one time step of the game logic given a list of actions.
        Returns the reward (or score change) obtained from performing the action(s).
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """
        Checks if the game is still in progress.
        """
        ...

    def state(self) -> np.ndarray:
        """
        Gets the current state of the game, processed or raw pixels
        """
        if self._use_processed_state:
            return self._processed_state()
        return self._raw_state()

    @abstractmethod
    def _processed_state(self) -> np.ndarray:
        """
        Gets the current state of the game, processed for the neural network.
        """
        ...

    @abstractmethod
    def _raw_state(self) -> np.ndarray:
        """
        Gets the current state of the game, but in raw pixels
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """
        Returns the game's name to be displayed
        """
        ...

    @abstractmethod
    def score(self) -> int:
        """
        Get current game's score
        """
        ...


GameType = TypeVar("GameType", bound=BaseGame, covariant=True)
