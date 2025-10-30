from collections import deque
import numpy as np
from typing import Optional
from .utils import Pos, Dir
from numpy.typing import NDArray


class Snake:
    def __init__(self, x: int, y: int, board_size: int):
        self._head: Pos = Pos(x, y)
        self._tail: deque[Pos] = deque([self._head])
        self._last_pos: Pos = Pos(x, y)
        self._board_size: int = board_size

        self._board: NDArray[np.int_] = np.full((board_size, board_size), 0)
        self._board[x, y] = 1

        self._dir = Dir.UP

    def head(self) -> Pos:
        return self._head

    def dir(self) -> Dir:
        return self._dir

    def length(self) -> int:
        return len(self._tail) + 1

    def tail(self) -> list[Pos]:
        return list(self._tail)

    def board(self) -> np.ndarray:
        return self._board

    def move(self) -> None:
        next_pos = (self._head + self._dir.vector()).mod_index(self._board_size)
        self._head = next_pos

        # Add new position
        self._tail.appendleft(self._head)
        self._board[*self._head] += 1

        # Remove last position
        self._last_pos = self._tail.pop()
        self._board[*self._last_pos] -= 1

    def turn(self, dir: Optional[Dir]) -> None:
        if dir is not None:
            if self._dir.opposite() == dir:
                return
            self._dir = dir

    def eat_food(self, food: Pos) -> bool:
        if food == self._head:
            self._tail.append(self._last_pos)
            self._board[*self._last_pos] += 1
            return True
        return False

    def collision(self) -> np.bool:
        return np.any(self._board.max() >= 2)
