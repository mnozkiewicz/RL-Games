from __future__ import annotations
import numpy as np
from .utils import Pos


class Board:
    def __init__(self, board: np.ndarray) -> None:
        assert board.shape == (23, 23), (
            "Board for pacman should be a square matrix of size (23, 23)"
        )
        self.board = board

    @classmethod
    def load(cls, path: str) -> Board:
        with open(path, "rb") as f:
            array: np.ndarray = np.load(f)
        return cls(array)

    def board_size(self) -> int:
        size: int = self.board.shape[0]
        return size

    def wall(self, pos: Pos) -> bool:
        return bool(self.board[*pos] == 1)

    def food(self, pos: Pos) -> bool:
        return bool(self.board[*pos] == 2)

    def eat_food(self, pos: Pos) -> bool:
        if self.food(pos):
            self.board[*pos] = 0
            return True
        else:
            return False


class BoardView:
    __slots__ = ("__board",)

    def __init__(self, board: Board) -> None:
        self.__board = board

    def size(self) -> int:
        return self.__board.board_size()

    def wall(self, pos: Pos) -> bool:
        return self.__board.wall(pos)

    def food(self, pos: Pos) -> bool:
        return self.__board.food(pos)
