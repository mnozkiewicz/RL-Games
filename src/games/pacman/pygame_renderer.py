import pygame
from collections import defaultdict
from typing import Dict
from ...utils.colors import Color
from ..base_pygame_renderer import BasePygameRenderer

# from pathlib import Path
from .game import PacmanGame


class PacmanPygameRenderer(BasePygameRenderer[PacmanGame]):
    def __init__(self, game: PacmanGame):
        super().__init__(game)

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        screen_width = surface.get_width()
        screen_height = surface.get_height()

        # Pixel size of single cell on the board
        self.cell_width = screen_width // self.game.board.shape[0]
        self.cell_height = screen_height // self.game.board.shape[0]

    def draw_rect(self, x: int, y: int) -> pygame.rect.Rect:
        return pygame.rect.Rect(
            x * self.cell_width, y * self.cell_height, self.cell_width, self.cell_height
        )

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.BLACK)

        for i in range(self.game.board.shape[0]):
            for j in range(self.game.board.shape[1]):
                if self.game.board[i, j] == 1:
                    pygame.draw.rect(surface, Color.WHITE, self.draw_rect(j, i))

        pacman_pos = self.game.pacman.get_pos()
        pygame.draw.rect(surface, Color.YELLOW, self.draw_rect(*reversed(pacman_pos)))

    def get_key_map(self) -> Dict[int, int]:
        return defaultdict(
            lambda: -1,
            {
                pygame.K_UP: 0,
                pygame.K_DOWN: 1,
                pygame.K_LEFT: 2,
                pygame.K_RIGHT: 3,
                pygame.K_w: 0,
                pygame.K_s: 1,
                pygame.K_a: 2,
                pygame.K_d: 3,
            },
        )
