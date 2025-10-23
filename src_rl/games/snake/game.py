import numpy as np
from .utils import Pos, State, Action
from .snake import Snake
from ..base_game import BaseGame

FOOD_REWARD = 30
DEATH_REWARD = -300
MOVE_REWARD = -1

label_to_action = {
    -1: Action.NOTHING,
    0: Action.UP,
    1: Action.DOWN,
    2: Action.LEFT,
    3: Action.RIGHT,
}


class SnakeGame(BaseGame):
    def __init__(self, board_size: int, infinite: bool = True):
        self.board_size: int = board_size
        self.infinte: bool = infinite

        self.snake: Snake
        self._current_state_index: int
        self._last_computed_state: int
        self._running: bool

        self._food: Pos
        self._state: State

        self.reset()

    def reset(self):
        self.snake = Snake(self.board_size // 2, self.board_size // 2, self.board_size)
        self._current_state_index = 0
        self._last_computed_state = 0
        self._running = True

        self.draw_new_food()
        self._state = self._compute_state()

    @property
    def number_of_moves(self) -> int:
        return 4

    def step(self, action_label: int) -> int:
        self._current_state_index += 1

        action = label_to_action[action_label]

        if action != Action.NOTHING:
            self.snake.turn(action.to_dir())

        self.snake.move()
        if self.snake.eat_food(self._food):
            self.draw_new_food()
            return FOOD_REWARD

        elif self.snake.collision():
            if not self.infinte:
                self._running = False
            else:
                self.reset()
            return DEATH_REWARD
        else:
            return MOVE_REWARD

    def get_state(self) -> State:
        if self._last_computed_state < self._current_state_index:
            self._state = self._compute_state()
        return self._state

    def is_running(self) -> bool:
        return self.get_state().running

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

    def draw_new_food(self) -> None:
        board = self.snake.board()
        vacant_places = board.size - board.sum().item()

        new_flattened_pos = np.random.randint(0, vacant_places)  # pyright: ignore

        false_indices = np.where(board.flatten() == 0)[0]
        flat_index = false_indices[new_flattened_pos]
        row, col = np.unravel_index(flat_index, board.shape)

        self._food = Pos(row.item(), col.item())

    def processed_state(self) -> np.ndarray:
        def get_local_window(state: State, window_size: int = 7):
            head_x, head_y = state.head
            food_x, food_y = state.food
            board_size = state.board_size

            half_w = window_size // 2
            window = np.zeros((window_size, window_size), dtype=np.float32)

            for i in range(window_size):
                for j in range(window_size):
                    x = (head_x - half_w + i) % board_size
                    y = (head_y - half_w + j) % board_size

                    if state.board[x, y] >= 1:
                        window[i, j] = 1.0  # body
                    elif (x, y) == (food_x, food_y):
                        window[i, j] = -1.0  # food
                    else:
                        window[i, j] = 0.0  # empty

            return window.flatten()

        state = self.get_state()
        head_x, head_y = state.head
        dir_x, dir_y = state.snake_dir.vector()
        board_size = state.board_size

        local_view = get_local_window(state, 7)

        def danger_in_direction(dx: int, dy: int):
            x = (head_x + dx) % board_size
            y = (head_y + dy) % board_size
            return 1.0 if state.board[x, y] >= 1 else 0.0

        food_x, food_y = state.food
        dx = (food_x - head_x + board_size) % board_size
        dy = (food_y - head_y + board_size) % board_size
        if dx > board_size / 2:
            dx -= board_size
        if dy > board_size / 2:
            dy -= board_size

        relational = np.array(
            [
                danger_in_direction(dir_x, dir_y),
                danger_in_direction(-dir_y, dir_x),
                danger_in_direction(dir_y, -dir_x),
                danger_in_direction(-dir_y, -dir_x),
                float(dir_x == 0 and dir_y == -1),  # up
                float(dir_x == 0 and dir_y == 1),  # down
                float(dir_x == -1 and dir_y == 0),  # left
                float(dir_x == 1 and dir_y == 0),  # right
                float(dy < 0),  # food up
                float(dy > 0),  # food down
                float(dx < 0),  # food left
                float(dx > 0),  # food right
            ],
            dtype=np.float32,
        )

        return np.concatenate((local_view, relational))
