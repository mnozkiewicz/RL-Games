import numpy as np
from ..base_game import BaseGame
from .utils import Action, Dir, Pos
from .pacman import Pacman
from .board import Board, BoardView
from .ghost import Ghost
from typing import Dict
from .map_prep.board import board_init


label_to_action = {
    -1: Action.NOTHING,
    0: Action.UP,
    1: Action.DOWN,
    2: Action.LEFT,
    3: Action.RIGHT,
}


class PacmanGame(BaseGame):
    FOOD_REWARD = 1
    MOVE_REWARD = 0
    DEATH_REWARD = -100
    WIN_REWARD = 100

    def __init__(self, infinite: bool = False, is_ai_controlled: bool = False) -> None:
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled

        self.__board: Board
        self.board_view: BoardView
        self.board_size: int

        self.pacman: Pacman
        self.ghosts: Dict[str, Ghost]

        self._queued_move: Dir
        self._running: bool
        self._score: int

        self.reset()

    def reset(self) -> None:
        self.__board = Board(board_init())
        self.board_view = BoardView(self.__board)
        self.board_size = self.__board.board_size()
        self.pacman = Pacman(Pos(2, 11), self.board_size)
        self.ghosts = {
            "cyan": Ghost("cyan", Pos(11, 9)),
            "orange": Ghost("orange", Pos(11, 10)),
            "pink": Ghost("pink", Pos(11, 12)),
            "red": Ghost("red", Pos(11, 13)),
        }

        self._running = True
        self._score = 0

    def end_episode(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def step(self, action_label: int) -> int:
        action = label_to_action[action_label]

        self.__board.update_pacman_pos(self.pacman.get_pos())
        self.__board.update_ghosts_pos(
            [ghost.get_pos() for _, ghost in self.ghosts.items()]
        )

        dir = action.to_dir()
        if dir is not None:
            self.pacman.change_dir(dir, self.board_view)

        self.pacman.step(self.board_view)
        self.__board.compute_shortest_paths()

        if self.__board.ghost(self.pacman.get_pos()):
            self.end_episode()
            return PacmanGame.DEATH_REWARD

        for _, ghost in self.ghosts.items():
            ghost.step(self.board_view)

        if self.__board.ghost(self.pacman.get_pos()):
            self.end_episode()
            return PacmanGame.DEATH_REWARD

        if self.__board.eat_food(self.pacman.get_pos()):
            self._score += 1
            return PacmanGame.FOOD_REWARD

        if self.__board.ghost(self.pacman.get_pos()):
            self.end_episode()
            return PacmanGame.DEATH_REWARD

        if self.__board.food_empty():
            self.end_episode()
            return PacmanGame.WIN_REWARD

        return PacmanGame.MOVE_REWARD

    def is_running(self) -> bool:
        return self._running

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return self._score

    def name(self) -> str:
        return "PACMAN"

    def processed_state(self) -> np.ndarray:
        return np.zeros([1, 2, 3], dtype=np.float32).flatten()
