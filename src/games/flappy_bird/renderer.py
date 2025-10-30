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
        return Image.fromarray(np.zeros((10, 10)))
