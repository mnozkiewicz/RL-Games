# from dataclasses import dataclass
# from .snake import Snake, Dir, Pos
# from enum import Enum
# from typing import Optional
# import numpy as np


# @dataclass
# class State:
#     board_size: int
#     head: Pos
#     tail: list[Pos]
#     food: Pos
#     snake_dir: Dir
#     board: np.ndarray
#     running: bool = True


# class Action(Enum):
#     QUIT: int = 0
#     UP: int = 1
#     DOWN: int = 2
#     LEFT: int = 3
#     RIGHT: int = 4
#     NOTHING: int = 5

#     def to_dir(self) -> Optional[Dir]:
#         match self:
#             case Action.UP:
#                 return Dir.UP
#             case Action.DOWN:
#                 return Dir.DOWN
#             case Action.LEFT:
#                 return Dir.LEFT
#             case Action.RIGHT:
#                 return Dir.RIGHT
#             case _:
#                 return None

#     @staticmethod
#     def move_actions() -> list['Action']:
#         return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


# class SnakeGame:

#     def __init__(self, board_size: int, infinite: bool = True):

#         self.board_size: int = board_size
#         self.infinte: bool = infinite

#         self.snake: Snake
#         self._current_state_index: int
#         self._last_computed_state: int
#         self._running: bool

#         self._food: Pos
#         self._state: State

#         self.reset_game()


#     def reset_game(self):
#         self.snake = Snake(self.board_size // 2, self.board_size // 2, self.board_size)
#         self._current_state_index = 0
#         self._last_computed_state = 0
#         self._running = True

#         self.draw_new_food()
#         self._state = self._compute_state()


#     def _compute_state(self) -> State:
#         self._last_computed_state = self._current_state_index
#         return State(
#             self.board_size,
#             self.snake.head(),
#             self.snake.tail(),
#             self._food,
#             self.snake.dir(),
#             self.snake.board(),
#             self._running
#         )

#     def state(self) -> State:
#         if self._last_computed_state < self._current_state_index:
#             self._state = self._compute_state()
#         return self._state

#     def draw_new_food(self) -> None:
#         board = self.snake._board
#         vacant_places = board.size - board.sum().item()
#         new_flattened_pos = np.random.randint(0, vacant_places)

#         false_indices = np.where(board.flatten() == False)[0]
#         flat_index = false_indices[new_flattened_pos]
#         row, col = np.unravel_index(flat_index, board.shape)

#         self._food = Pos(row.item(), col.item())


#     def game_step(self, actions: list[Action]):
#         self._current_state_index += 1

#         if actions:
#             action = actions[-1]
#             self.snake.turn(action.to_dir())

#         self.snake.move()
#         if self.snake.eat_food(self._food):
#             self.draw_new_food()

#         if self.snake.collision():
#             if not self.infinte:
#                 self._running = False
#             else:
#                 self.reset_game()

