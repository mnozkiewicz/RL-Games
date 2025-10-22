import numpy as np
from .utils import Pos, State, Action
from .snake import Snake
from ..base_game_engine import BaseGameEngine
from ..colors import Color
import pygame


action_to_label = {Action.UP: 0, Action.DOWN: 1, Action.LEFT: 2, Action.RIGHT: 3}
label_to_action = {v: k for k, v in action_to_label.items()}

FOOD_REWARD = 10
DEATH_REWARD = -100
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
        if event.type == pygame.KEYUP and (action := key_to_action_map(event.key)):
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
