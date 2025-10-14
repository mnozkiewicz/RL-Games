from abc import ABC, abstractmethod


class BaseGameEngine(ABC):
    @abstractmethod
    def reset(self):
        """Reset the game to initial state."""
        pass

    @abstractmethod
    def step(self, actions: list[int]):
        """Apply actions and update game state."""
        pass

    @abstractmethod
    def get_state(self):
        """Return current state."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if game is still running."""
        pass

    @abstractmethod
    def legal_actions(self):
        """Return list of legal actions."""
        pass
