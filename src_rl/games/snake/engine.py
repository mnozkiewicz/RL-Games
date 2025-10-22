from PIL import Image
import numpy as np
from .utils import Pos, State, Action
from .snake import Snake
from ..base_game import BaseGameEngine
from ...utils.colors import Color
import pygame
from typing import Optional


action_to_label = {Action.UP: 0, Action.DOWN: 1, Action.LEFT: 2, Action.RIGHT: 3}
label_to_action = {v: k for k, v in action_to_label.items()}

FOOD_REWARD = 30
DEATH_REWARD = -300
MOVE_REWARD = -1


def key_to_action_map(event_type: pygame.event.Event) -> Action:
    match event_type:
        case pygame.K_UP | pygame.K_w:
            return Action.UP
        case pygame.K_DOWN | pygame.K_s:
            return Action.DOWN
        case pygame.K_LEFT | pygame.K_a:
            return Action.LEFT
        case pygame.K_RIGHT | pygame.K_d:
            return Action.RIGHT
        case _:
            return Action.NOTHING


def map_events(events: list[pygame.event.Event]) -> list[Action]:
    actions: list[Action] = []
    for event in events:
        if event.type == pygame.KEYDOWN and (action := key_to_action_map(event.key)):
            actions.append(action)

    return actions


class SnakeGame(BaseGameEngine):
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

    def parse_user_input(self, events: list[pygame.event.Event]) -> list[int]:
        actions = map_events(events)
        action_labels = [
            action_to_label[action] for action in actions if action != Action.NOTHING
        ]
        return action_labels

    @property
    def number_of_moves(self) -> int:
        return 4

    def step(self, actions: list[int]) -> int:
        self._current_state_index += 1

        if actions:
            action = actions[-1]
            self.snake.turn(label_to_action[action].to_dir())

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

    def game_screenshot(self, output_size: Optional[int] = None) -> Image.Image:
        if output_size is not None and output_size <= 0:
            raise ValueError(
                f"`output_size should be a positive integer, given: {output_size}"
            )

        state = self.get_state()
        size = state.board_size
        screenshot = np.full((size, size, 3), fill_value=255, dtype=np.uint8)

        for tail_pos in state.tail:
            x, y = tail_pos
            screenshot[x, y] = Color.GREEN

        food_x, food_y = state.food
        screenshot[food_x, food_y] = Color.BLUE

        head_x, head_y = state.head
        screenshot[head_x, head_y] = Color.RED

        img = Image.fromarray(screenshot)
        if output_size is None:
            return img

        img = img.resize((output_size, output_size), resample=Image.Resampling.NEAREST)
        return img

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

    def create_rect(
        self, x: int, y: int, cell_width: int, cell_height: int
    ) -> pygame.Rect:
        return pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

    def draw_state(self, screen: pygame.Surface):
        state = self.get_state()

        if not state.running:
            screen.fill(Color.BLACK)
            return

        screen_width = screen.get_width()
        screen_height = screen.get_height()

        cell_width = screen_width // self.board_size
        cell_height = screen_height // self.board_size

        screen.fill(Color.WHITE)
        for pos in state.tail:
            rect = self.create_rect(pos.x, pos.y, cell_width, cell_height)
            pygame.draw.rect(screen, Color.GREEN, rect)

        head = state.head
        rect = self.create_rect(head.x, head.y, cell_width, cell_height)
        pygame.draw.rect(screen, Color.RED, rect)

        food = state.food
        rect = self.create_rect(food.x, food.y, cell_width, cell_height)
        pygame.draw.rect(screen, Color.BLUE, rect)

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
