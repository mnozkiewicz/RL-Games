import pygame
from src_rl.utils.colors import Color
from .game import SnakeGame
from ..base_renderer import BaseRenderer
from PIL import Image
from typing import Optional
import numpy as np


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
