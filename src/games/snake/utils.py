from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from typing import Generator


@dataclass(frozen=True, eq=True)
class Pos:
    x: int
    y: int

    def __add__(self, other: Pos) -> Pos:
        return Pos(self.x + other.x, self.y + other.y)

    def mod_index(self, size: int) -> Pos:
        return Pos(self.x % size, self.y % size)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.x
        yield self.y


class Dir(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    def vector(self) -> Pos:
        match self:
            case Dir.UP:
                return Pos(0, -1)
            case Dir.DOWN:
                return Pos(0, 1)
            case Dir.LEFT:
                return Pos(-1, 0)
            case Dir.RIGHT:
                return Pos(1, 0)

    def opposite(self) -> Dir:
        match self:
            case Dir.UP:
                return Dir.DOWN
            case Dir.DOWN:
                return Dir.UP
            case Dir.LEFT:
                return Dir.RIGHT
            case Dir.RIGHT:
                return Dir.LEFT


class Action(Enum):
    QUIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    NOTHING = 5

    def to_dir(self) -> Optional[Dir]:
        match self:
            case Action.UP:
                return Dir.UP
            case Action.DOWN:
                return Dir.DOWN
            case Action.LEFT:
                return Dir.LEFT
            case Action.RIGHT:
                return Dir.RIGHT
            case _:
                return None


@dataclass
class State:
    board_size: int
    head: Pos
    tail: list[Pos]
    food: Pos
    snake_dir: Dir
    board: NDArray[np.int_]
    running: bool = True
