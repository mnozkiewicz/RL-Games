import numpy as np
from ..base_game import BaseGame
from .utils import Action, Dir
from .pacman import Pacman


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

        self.pacman: Pacman
        self._queued_move: Dir
        self._running: bool
        self._score: int

        self.reset()

    def reset(self) -> None:
        self.pacman = Pacman((2, 11), self.board_size)
        self._queued_move = self.pacman.get_dir()
        self._running = True
        self._score = 0

    def step(self, action_label: int) -> int:
        action = label_to_action[action_label]

        dir = action.to_dir()
        if dir is not None:
            self._queued_move = dir

        dir = self._queued_move
        pacman_dir = self.pacman.get_dir()

        if dir != pacman_dir:
            cur_x, cur_y = self.pacman.get_pos()
            shift_x, shift_y = dir.to_vector()
            new_x, new_y = (
                (cur_x + shift_x) % self.board_size,
                (cur_y + shift_y) % self.board_size,
            )

            if self.board[new_x, new_y]:
                pass

        return PacmanGame.REWARD

    def is_running(self) -> bool:
        return True

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return self._score

    def name(self) -> str:
        return "PACMAN"

    def processed_state(self) -> np.ndarray:
        return np.zeros([1, 2, 3], dtype=np.float32).flatten()
