from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
from .utils import Pos, Dir
from collections import deque


class Board:
    def __init__(self, board: np.ndarray) -> None:
        assert board.shape == (23, 23), (
            "Board for pacman should be a square matrix of size (23, 23)"
        )
        self.board = board
        self.pacman_pos: Optional[Pos] = None
        self.ghosts_pos: Optional[List[Pos]] = None
        self.shortest_paths: Optional[Dict[Pos, Dir]] = None

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

    def ghost(self, pos: Pos) -> bool:
        if self.ghosts_pos is None:
            return False

        for ghost_pos in self.ghosts_pos:
            if pos == ghost_pos:
                return True
        return False

    def update_pacman_pos(self, pos: Pos) -> None:
        self.pacman_pos = pos

    def update_ghosts_pos(self, ghosts: List[Pos]) -> None:
        self.ghosts_pos = []
        for ghost_pos in ghosts:
            self.ghosts_pos.append(ghost_pos)

    def compute_shortest_paths(self) -> None:
        if self.pacman_pos is None:
            return

        queue = deque([self.pacman_pos])
        parent: dict[Pos, Dir] = {queue[0]: Dir.UP}
        size = self.board_size()
        while queue:
            pos = queue.popleft()
            for dir in Dir:
                new_pos = (pos + dir.vector()).mod_index(size)
                if new_pos not in parent and not self.wall(new_pos):
                    parent[new_pos] = dir.opposite()
                    queue.append(new_pos)

        self.shortest_paths = parent

    def get_shortest_path(self, pos: Pos) -> Optional[Dir]:
        if self.shortest_paths is None:
            return None

        if pos not in self.shortest_paths:
            return None

        return self.shortest_paths[pos]

    def food_empty(self) -> bool:
        food_count = (self.board == 2).sum()
        return bool(food_count <= 0)


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

    def ghost(self, pos: Pos) -> bool:
        return self.__board.ghost(pos)

    def get_shortest_path(self, pos: Pos) -> Optional[Dir]:
        return self.__board.get_shortest_path(pos)
