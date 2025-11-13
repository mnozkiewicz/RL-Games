import numpy as np
from .utils import Pos, State, Action
from .snake import Snake
from ..base_game import BaseGame

label_to_action = {
    -1: Action.NOTHING,
    0: Action.UP,
    1: Action.DOWN,
    2: Action.LEFT,
    3: Action.RIGHT,
}


class SnakeGame(BaseGame):
    FOOD_REWARD = 10.0
    DEATH_REWARD = -500.0
    MOVE_REWARD = -0.1

    def __init__(
        self, board_size: int, infinite: bool = True, is_ai_controlled: bool = False
    ) -> None:
        self.board_size: int = board_size
        self.infinite: bool = infinite
        self.is_ai_controlled = is_ai_controlled

        self.snake: Snake
        self._current_state_index: int
        self._last_computed_state: int
        self._running: bool

        self._food: Pos
        self._state: State
        self._score: int

        self.reset()

    def reset(self) -> None:
        self.snake = Snake(self.board_size // 2, self.board_size // 2, self.board_size)
        self._current_state_index = 0
        self._last_computed_state = 0
        self._running = True
        self._score = 0

        self.draw_new_food()
        self._state = self._compute_state()

    @property
    def number_of_moves(self) -> int:
        return 4

    def name(self) -> str:
        return "SNAKE"

    def score(self) -> int:
        return self._score

    def end_game(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def step(self, action_label: int) -> float:
        if not self._running:
            return SnakeGame.DEATH_REWARD

        self._current_state_index += 1

        action = label_to_action[action_label]

        if action != Action.NOTHING:
            self.snake.turn(action.to_dir())

        wall_collision = self.snake.move()
        if wall_collision:
            self.end_game()
            return SnakeGame.DEATH_REWARD

        if self.snake.eat_food(self._food):
            self._score += 1
            self.draw_new_food()
            return SnakeGame.FOOD_REWARD

        elif self.snake.collision():
            self.end_game()
            return SnakeGame.DEATH_REWARD
        else:
            return SnakeGame.MOVE_REWARD

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
        def get_local_window(state: State, window_size: int = 7) -> np.ndarray:
            head_x, head_y = state.head
            food_x, food_y = state.food
            board_size = state.board_size

            half_w = window_size // 2
            window = np.zeros((window_size, window_size), dtype=np.float32)

            for i in range(window_size):
                for j in range(window_size):
                    x = head_x - half_w + i
                    y = head_y - half_w + j

                    # Check if this position is outside the board
                    if x < 0 or x >= board_size or y < 0 or y >= board_size:
                        window[i, j] = 2.0  # wall
                    elif state.board[x, y] >= 1:
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

        local_view = get_local_window(state, 5)

        def danger_in_direction(dx: int, dy: int) -> float:
            x = head_x + dx
            y = head_y + dy

            # Check wall or body collision
            if x < 0 or x >= board_size or y < 0 or y >= board_size:
                return 1.0  # wall danger
            elif state.board[x, y] >= 1:
                return 1.0  # body danger
            else:
                return 0.0

        food_x, food_y = state.food
        dx = food_x - head_x
        dy = food_y - head_y

        relational = np.array(
            [
                float(head_x) / board_size,  # Head pos
                float(head_y) / board_size,
                len(state.tail),  # Snake length
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

        return np.concatenate((relational, local_view))

    def entire_state(self) -> np.ndarray:
        state = np.zeros((1, *self.snake.board().shape), dtype=np.float32)

        for pos in self.snake.tail():
            state[0][*pos] = 1.0

        state[0][*self.snake.head()] = 1.0
        state[0][*self._food] = -1.0
        return state
