from .utils import Pos
from .utils import Dir
from .board import BoardView
from typing import List
from random import choice, uniform


class Ghost:
    def __init__(self, color: str, pos: Pos):
        self.color = color
        self._pos = pos
        self._dir = Dir.LEFT

    def get_pos(self) -> Pos:
        return self._pos

    def get_dir(self) -> Dir:
        return self._dir

    def compute_next_pos(self, dir: Dir, view: BoardView) -> Pos:
        return (self._pos + dir.vector()).mod_index(view.size())

    def step(self, view: BoardView) -> None:
        possible_moves: List[Dir] = []

        for dir in Dir:
            if (
                not view.wall(self.compute_next_pos(dir, view))
                and dir.opposite() != self._dir
            ):
                possible_moves.append(dir)

        to_pacman = view.get_shortest_path(self._pos)
        if len(possible_moves) > 0:
            if to_pacman in possible_moves and uniform(0.0, 1.0) > 0.8:
                self._dir = to_pacman
            else:
                self._dir = choice(possible_moves)

        next_pos = self.compute_next_pos(self._dir, view)
        self._pos = next_pos
