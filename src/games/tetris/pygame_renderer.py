import pygame
from .game import TetrisGame
from ...utils.colors import Color
from ..base_pygame_renderer import BasePygameRenderer

# from pathlib import Path
from typing import Dict
from collections import defaultdict


class TetrisPygameRenderer(BasePygameRenderer[TetrisGame]):
    def __init__(self, game: TetrisGame):
        super().__init__(game)

    # def resize(self, surface: pygame.Surface, size: Tuple[int, int]) -> pygame.Surface:
    #     return pygame.transform.scale(surface, size).convert_alpha()

    # def load_image(
    #     self, path: Path, size: Optional[Tuple[int, int]] = None
    # ) -> pygame.Surface:
    #     image = pygame.image.load(path).convert_alpha()
    #     if size is None:
    #         return image
    #     return self.resize(image, size)

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        self.screen_width = surface.get_width()
        self.screen_height = surface.get_height()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.LIGHTGREEN)

    def get_key_map(self) -> Dict[int, int]:
        return defaultdict(
            lambda: -1,
            {
                pygame.K_DOWN: 0,
                pygame.K_UP: 1,
                pygame.K_s: 0,
                pygame.K_w: 1,
            },
        )
