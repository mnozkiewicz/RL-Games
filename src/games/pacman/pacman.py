from .utils import Dir, Pos
from .board import BoardView
from typing import Optional


class Pacman:
    def __init__(self, start_pos: Pos, board_size: int):
        self._pos = start_pos
        self._dir = Dir.UP
        self.board_size = board_size
        self._queued_move: Optional[Dir] = None

    def get_pos(self) -> Pos:
        return self._pos

    def get_dir(self) -> Dir:
        return self._dir

    def compute_next_pos(self, dir: Optional[Dir] = None) -> Pos:
        if dir is None:
            dir = self._dir
        return (self._pos + dir.vector()).mod_index(self.board_size)

    def change_dir(self, dir: Dir, board: BoardView) -> None:
        self._queued_move = None

        next_pos = self.compute_next_pos(dir)
        if not board.wall(next_pos):
            self._dir = dir
        else:
            self._queued_move = dir

    def step(self, board: BoardView) -> None:
        if self._queued_move is not None:
            self.change_dir(self._queued_move, board)

        next_pos = self.compute_next_pos()
        if not board.wall(next_pos):
            self._pos = next_pos
