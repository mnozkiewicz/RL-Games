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

    def compute_next_pos(self, dir: Dir, view: BoardView) -> Pos:
        return (self._pos + dir.vector()).mod_index(view.size())

    def step(self, view: BoardView) -> None:
        possible_moves: List[Dir] = []

        for dir in Dir:
            if not view.wall(self.compute_next_pos(dir, view)):
                possible_moves.append(dir)

        to_pacman = view.get_shortest_path(self._pos)

        new_dir = self._dir
        if (
            to_pacman is not None
            and to_pacman in possible_moves
            and uniform(0.0, 1.0) > 0.5
        ):
            new_dir = to_pacman
        elif len(possible_moves) > 0:
            new_dir = choice(possible_moves)

        if new_dir != self._dir.opposite():
            self._dir = new_dir

        next_pos = self.compute_next_pos(self._dir, view)
        self._pos = next_pos
