import numpy as np
from .utils import Pos, State, Action
from .snake import Snake


class SnakeGame:
    def __init__(self, board_size: int, infinite: bool = True):
        self.board_size: int = board_size
        self.infinte: bool = infinite

        self.snake: Snake
        self._current_state_index: int
        self._last_computed_state: int
        self._running: bool

        self._food: Pos
        self._state: State

        self.reset_game()

    def reset_game(self):
        self.snake = Snake(self.board_size // 2, self.board_size // 2, self.board_size)
        self._current_state_index = 0
        self._last_computed_state = 0
        self._running = True

        self.draw_new_food()
        self._state = self._compute_state()

    def _compute_state(self) -> State:
        self._last_computed_state = self._current_state_index
        return State(
            self.board_size,
            self.snake.head(),
            self.snake.tail(),
            self._food,
            self.snake.dir(),
            self.snake.board(),
            self._running,
        )

    def state(self) -> State:
        if self._last_computed_state < self._current_state_index:
            self._state = self._compute_state()
        return self._state

    def draw_new_food(self) -> None:
        board = self.snake.board()
        vacant_places = board.size - board.sum().item()

        new_flattened_pos = np.random.randint(0, vacant_places)  # pyright: ignore

        false_indices = np.where(~board.flatten())[0]
        flat_index = false_indices[new_flattened_pos]
        row, col = np.unravel_index(flat_index, board.shape)

        self._food = Pos(row.item(), col.item())

    def game_step(self, actions: list[Action]):
        self._current_state_index += 1

        if actions:
            action = actions[-1]
            self.snake.turn(action.to_dir())

        self.snake.move()
        if self.snake.eat_food(self._food):
            self.draw_new_food()

        if self.snake.collision():
            if not self.infinte:
                self._running = False
            else:
                self.reset_game()
