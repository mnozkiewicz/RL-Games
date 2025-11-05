from typing import Tuple
from .utils import Dir


class Pacman:
    def __init__(self, start_pos: Tuple[int, int], board_size: int):
        self._x = start_pos[0]
        self._y = start_pos[1]

        self._dir = Dir.UP
        self.board_size = board_size

    def get_pos(self) -> Tuple[int, int]:
        return self._x, self._y

    def get_dir(self) -> Dir:
        return self._dir

    def change_dir(self, dir: Dir) -> None:
        self._dir = dir

    def move(self) -> None:
        shift_x, shift_y = self._dir.to_vector()
        self._x = (self._x + shift_x) % self.board_size
        self._y = (self._y + shift_y) % self.board_size
