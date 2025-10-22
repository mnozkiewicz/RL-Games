import pygame
from pygame.event import Event
from ...utils.colors import Color
from .game import SnakeGame, action_to_label
from ..base_renderer import BaseRenderer
from PIL import Image
from typing import Optional
import numpy as np
from .utils import Action


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


class SnakeRenderer(BaseRenderer[SnakeGame]):
    def __init__(self, game: SnakeGame):
        super().__init__(game)
        self.head_color = Color.RED
        self.tail_color = Color.GREEN
        self.food_color = Color.BLUE

    def create_rect(
        self, x: int, y: int, cell_width: int, cell_height: int
    ) -> pygame.Rect:
        return pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

    def draw(self, surface: pygame.Surface) -> None:
        state = self.game.get_state()

        if not state.running:
            surface.fill(Color.BLACK)
            return

        screen_width = surface.get_width()
        screen_height = surface.get_height()

        cell_width = screen_width // self.game.board_size
        cell_height = screen_height // self.game.board_size

        surface.fill(Color.WHITE)
        for pos in state.tail:
            rect = self.create_rect(pos.x, pos.y, cell_width, cell_height)
            pygame.draw.rect(surface, self.tail_color, rect)

        head = state.head
        rect = self.create_rect(head.x, head.y, cell_width, cell_height)
        pygame.draw.rect(surface, self.head_color, rect)

        food = state.food
        rect = self.create_rect(food.x, food.y, cell_width, cell_height)
        pygame.draw.rect(surface, self.food_color, rect)

    def game_screenshot(self, output_size: Optional[int] = None) -> Image.Image:
        if output_size is not None and output_size <= 0:
            raise ValueError(
                f"`output_size should be a positive integer, given: {output_size}"
            )

        state = self.game.get_state()
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

    def handle_events(self, events: list[Event]) -> list[int]:
        actions = map_events(events)
        action_labels = [
            action_to_label[action] for action in actions if action != Action.NOTHING
        ]
        return action_labels
