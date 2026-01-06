import pygame
from .game import TetrisGame
from ...utils.colors import Color
from ..base_pygame_renderer import BasePygameRenderer

# from pathlib import Path
from typing import Dict
import numpy as np
from collections import defaultdict


class TetrisPygameRenderer(BasePygameRenderer[TetrisGame]):
    def __init__(self, game: TetrisGame):
        super().__init__(game)

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        self.screen_width = surface.get_width()
        self.screen_height = surface.get_height()

        self.board_height, self.board_width = self.game.board.shape
        assert self.board_height // 2 == self.board_width, (
            "Height of a board for tetris should be double the width"
        )

        self.game_board_height = self.screen_height
        self.game_board_width = self.screen_height // 2
        self.cell_size = self.game_board_width // self.board_width

        self.padding = (self.screen_width - self.game_board_width) // 2
        self.game_rect = pygame.Rect(
            self.padding, 0, self.game_board_width, self.screen_height
        )

    def _draw_grid_lines(self, surface: pygame.Surface) -> None:
        for i in range(self.board_width + 1):
            x_coord = self.padding + i * self.cell_size
            pygame.draw.line(
                surface=surface,
                color=Color.BLACK,
                start_pos=(x_coord, 0),
                end_pos=(x_coord, self.screen_height),
                width=1,
            )

        for i in range(self.board_height + 1):
            y_coord = i * self.cell_size
            pygame.draw.line(
                surface=surface,
                color=Color.BLACK,
                start_pos=(self.padding, y_coord),
                end_pos=(self.padding + self.game_board_width, y_coord),
                width=1,
            )

    def _draw_cur_shape(self, surface: pygame.Surface) -> None:
        shape = self.game.shape_manager.current_shape()
        mask = shape.mask()

        x, y = shape.x, shape.y

        for (i_offset, j_offset), value in np.ndenumerate(mask):
            if value <= 0:
                continue

            i, j = x + i_offset, y + j_offset
            pygame.draw.rect(
                surface=surface,
                color=Color.RED,
                rect=(
                    self.padding + j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

    def _draw_static_shapes(self, surface: pygame.Surface) -> None:
        for (i, j), value in np.ndenumerate(self.game.board):
            if value <= 0:
                continue

            pygame.draw.rect(
                surface=surface,
                color=Color.RED,
                rect=(
                    self.padding + j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.LIGHTGREEN)
        surface.fill(Color.LIGHTBLUE, self.game_rect)

        self._draw_cur_shape(surface)
        self._draw_grid_lines(surface)
        self._draw_static_shapes(surface)

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
