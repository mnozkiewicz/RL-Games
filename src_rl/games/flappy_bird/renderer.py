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

    # def init_renderer():

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.WHITE)

        screen_width = surface.get_width()
        screen_height = surface.get_height()

        flappy_x = int(screen_width * self.game.x)
        flappy_y = int(screen_height * self.game.y)
        height = int(screen_height * self.game.size)

        rect = pygame.Rect(flappy_x, flappy_y, height, height)
        pygame.draw.rect(surface, Color.RED, rect)

    def game_screenshot(self, output_size: Optional[int] = None) -> Image.Image:
        return Image.fromarray(np.zeros((10, 10)))
