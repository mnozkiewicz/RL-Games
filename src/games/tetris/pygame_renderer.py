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

        self.board_height, self.board_width = self.game.board.shape
        assert self.board_height // 2 == self.board_width, (
            "Height of a board for tetris should be double the width"
        )

        padding = (self.screen_width - (self.screen_height // 2)) // 2
        self.game_rect = pygame.Rect(
            padding, 0, self.screen_height // 2, self.screen_height
        )
        # surface.fill(Color.LIGHTGREEN, rect)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.LIGHTGREEN)
        surface.fill(Color.LIGHTBLUE, self.game_rect)

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
