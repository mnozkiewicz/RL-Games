# from collections import deque
# from dataclasses import dataclass
# from enum import Enum
# import numpy as np
# from typing import Optional


# @dataclass(frozen=True, eq=True)
# class Pos:
#     x: int
#     y: int

#     def __add__(self, other: 'Pos') -> 'Pos':
#         return Pos(self.x + other.x, self.y + other.y)

#     def mod_index(self, size: int) -> 'Pos':
#         return Pos(self.x % size, self.y % size)

#     def __iter__(self):
#         yield self.x
#         yield self.y


# class Dir(Enum):
#     UP: str = 'UP'
#     DOWN: str = 'DOWN'
#     LEFT: str = 'LEFT'
#     RIGHT: str = 'RIGHT'

#     def vector(self) -> Pos:
#         match self:
#             case Dir.UP:
#                 return Pos(0, -1)
#             case Dir.DOWN:
#                 return Pos(0, 1)
#             case Dir.LEFT:
#                 return Pos(-1, 0)
#             case Dir.RIGHT:
#                 return Pos(1, 0)
#             case _:
#                 pass

#     def opposite(self) -> 'Dir':
#         match self:
#             case Dir.UP:
#                 return Dir.DOWN
#             case Dir.DOWN:
#                 return Dir.UP
#             case Dir.LEFT:
#                 return Dir.RIGHT
#             case Dir.RIGHT:
#                 return Dir.LEFT

# class Snake:
#     def __init__(self, x: int, y: int, board_size: int):

#         self._head: Pos = Pos(x, y)
#         self._tail: deque[Pos] = deque([self._head])
#         self._last_pos: Pos = Pos(x, y)
#         self._board_size: int = board_size

#         self._board: np.ndarray[np.bool] = np.full((board_size, board_size), 0)
#         self._board[x, y] = 1

#         self._dir = Dir.UP

#     def head(self) -> Pos:
#         return self._head

#     def dir(self) -> Dir:
#         return self._dir

#     def length(self) -> int:
#         return len(self._tail) + 1

#     def tail(self) -> list[Pos]:
#         return list(self._tail)

#     def board(self) -> np.ndarray:
#         return self._board

#     def move(self) -> None:
#         next_pos = (self._head + self._dir.vector()).mod_index(self._board_size)
#         self._head = next_pos

#         # Add new position
#         self._tail.appendleft(self._head)
#         self._board[*self._head] += 1

#         # Remove last position
#         self._last_pos = self._tail.pop()
#         self._board[*self._last_pos] -= 1

#     def turn(self, dir: Optional[Dir]) -> None:
#         if dir is not None:
#             if self._dir.opposite() == dir:
#                 return
#             self._dir = dir

#     def eat_food(self, food: Pos) -> bool:
#         if food == self._head:
#             self._tail.append(self._last_pos)
#             self._board[*self._last_pos] += 1
#             return True
#         return False


#     def collision(self) -> bool:
#         return np.any(self._board.max() >= 2)
