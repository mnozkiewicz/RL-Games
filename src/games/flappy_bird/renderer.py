import pygame
from ...utils.colors import Color
from .game import FlappyGame
from ..base_renderer import BaseRenderer
from PIL import Image
from typing import Optional
import numpy as np


class FlappyRenderer(BaseRenderer[FlappyGame]):
    def __init__(self, game: FlappyGame):
        super().__init__(game)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.WHITE)

        screen_width = surface.get_width()
        screen_height = surface.get_height()

        flappy_x = int(screen_width * self.game.bird.x)
        flappy_y = int(screen_height * self.game.bird.y)
        height = int(screen_height * self.game.bird.size)

        rect = pygame.Rect(flappy_x, flappy_y, height, height)
        pygame.draw.rect(surface, Color.RED, rect)

        for obstacle in self.game.obstacles:
            upper_rect = pygame.Rect(
                obstacle.x * screen_width,
                0,
                obstacle.width * screen_width,
                obstacle.top * screen_height,
            )

            lower_rect = pygame.Rect(
                obstacle.x * screen_width,
                obstacle.bottom * screen_height,
                obstacle.width * screen_width,
                (1 - obstacle.bottom) * screen_height,
            )

            pygame.draw.rect(surface, Color.GREEN, upper_rect)
            pygame.draw.rect(surface, Color.GREEN, lower_rect)

    def game_screenshot(self, output_size: Optional[int] = None) -> Image.Image:
        if output_size is not None and output_size <= 0:
            raise ValueError(
                f"`output_size should be a positive integer, given: {output_size}"
            )
        if output_size is None:
            output_size = 256

        screenshot = np.full(
            (output_size, output_size, 3), fill_value=255, dtype=np.uint8
        )

        flappy_x = int(output_size * self.game.bird.x)
        flappy_y = int(output_size * self.game.bird.y)
        height = int(output_size * self.game.bird.size)

        screenshot[flappy_y : flappy_y + height, flappy_x : flappy_x + height] = (
            Color.RED
        )

        for obstacle in self.game.obstacles:
            left = int(obstacle.x * output_size)
            right = int((obstacle.x + obstacle.width) * output_size)
            top = int(obstacle.top * output_size)
            bottom = int(obstacle.bottom * output_size)

            screenshot[:top, left:right] = Color.GREEN
            screenshot[bottom:, left:right] = Color.GREEN

        return Image.fromarray(screenshot)
