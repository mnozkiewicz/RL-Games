from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple


class Dir(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    def to_vector(self) -> Tuple[int, int]:
        match self:
            case Dir.UP:
                return (-1, 0)
            case Dir.DOWN:
                return (1, 0)
            case Dir.LEFT:
                return (0, -1)
            case Dir.RIGHT:
                return (0, 1)

    def angle(self) -> int:
        match self:
            case Dir.UP:
                return 90
            case Dir.DOWN:
                return -90
            case Dir.LEFT:
                return 180
            case Dir.RIGHT:
                return 0

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
