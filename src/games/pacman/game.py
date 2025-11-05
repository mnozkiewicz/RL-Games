import numpy as np
from ..base_game import BaseGame
from .utils import Action


label_to_action = {
    -1: Action.NOTHING,
    0: Action.UP,
    1: Action.DOWN,
    2: Action.LEFT,
    3: Action.RIGHT,
}


def load_board(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        array: np.ndarray = np.load(f)
    return array


class PacmanGame(BaseGame):
    REWARD = 0

    def __init__(self, infinite: bool = False) -> None:
        self.infinite = infinite
        self.board = load_board("assets/pacman/board.npy")
        self.board_size = self.board.shape[0]
        self.reset()

    def reset(self) -> None:
        pass

    @property
    def number_of_moves(self) -> int:
        return 4

    def name(self) -> str:
        return "PACMAN"

    def score(self) -> int:
        return 0

    def step(self, action_label: int) -> int:
        return PacmanGame.REWARD

    def is_running(self) -> bool:
        return True

    def processed_state(self) -> np.ndarray:
        return np.zeros([1, 2, 3], dtype=np.float32).flatten()
