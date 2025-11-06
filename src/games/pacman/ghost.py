from .utils import Pos
from .board import BoardView
from .utils import Dir
from random import choice


class Ghost:
    def __init__(self, color: str, pos: Pos):
        self.color = color
        self._pos = pos
        self._dir = Dir.LEFT

    def get_pos(self) -> Pos:
        return self._pos

    def compute_next_pos(self, view: BoardView) -> Pos:
        return (self._pos + self._dir.vector()).mod_index(view.size())

    def step(self, view: BoardView) -> None:
        self._dir = choice(list(Dir))
        next_pos = self.compute_next_pos(view)
        if not view.wall(next_pos):
            self._pos = next_pos
