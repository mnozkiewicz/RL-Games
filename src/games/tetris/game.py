import numpy as np
from ..base_game import BaseGame


class TetrisGame(BaseGame):
    def __init__(self, infinite: bool = True, is_ai_controlled: bool = False) -> None:
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled

        self._running: bool = True
        self.reset()

    def reset(self) -> None:
        self._score = 0
        self._running = True

    def step(self, action_label: int) -> float:
        return 5

    def is_running(self) -> bool:
        return self._running

    def processed_state(self) -> np.ndarray:
        return np.array([[0]], dtype=np.float32)

    def name(self) -> str:
        return "Tetris"

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return 0
