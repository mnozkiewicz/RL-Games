import pygame
from collections import defaultdict
from typing import Dict

from .utils import Pos
from ...utils.colors import Color
from ..base_pygame_renderer import BasePygameRenderer

from pathlib import Path
from .game import PacmanGame


class PacmanPygameRenderer(BasePygameRenderer[PacmanGame]):
    def __init__(self, game: PacmanGame):
        super().__init__(game)

    def load_image(self, path: Path) -> pygame.Surface:
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (self.cell_width, self.cell_height))

    def rotate_pacman(self, image: pygame.Surface, angle: int) -> pygame.Surface:
        rotated = pygame.transform.rotate(image, angle)
        if angle == 180:
            return pygame.transform.flip(rotated, flip_x=0, flip_y=1)
        return rotated

    def load_assets(self) -> None:
        # Path to folder containing assets
        asset_path = Path("assets/pacman")

        if not asset_path.exists():
            raise FileNotFoundError(
                f"Could not locate assets for pacman under: {asset_path}"
                f"Check the 'Downloading assets and weights' section in README.md"
            )

        self.dot = self.load_image(asset_path / "dot.png")

        self.ghosts: Dict[str, pygame.Surface] = {
            color: self.load_image(asset_path / "ghosts" / f"{color}.png")
            for color in self.game.ghosts.keys()
        }

        self.pacman: Dict[int, pygame.Surface] = {
            0: self.load_image(asset_path / "pacman-1.png"),
            1: self.load_image(asset_path / "pacman-2.png"),
            2: self.load_image(asset_path / "pacman-3.png"),
        }

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        screen_width = surface.get_width()
        screen_height = surface.get_height()

        # Pixel size of single cell on the board
        self.cell_width = screen_width // self.game.board_size
        self.cell_height = screen_height // self.game.board_size

        self.pacman_state = 2
        self.load_assets()

    def rect(self, x: int, y: int) -> pygame.rect.Rect:
        return pygame.rect.Rect(
            x * self.cell_width, y * self.cell_height, self.cell_width, self.cell_height
        )

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.BLUE)

        self.pacman_state = (self.pacman_state + 1) % 3

        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                pos = Pos(i, j)
                if self.game.board_view.wall(pos):
                    pygame.draw.rect(surface, Color.BLACK, self.rect(j, i))
                if self.game.board_view.food(pos):
                    surface.blit(self.dot, (j * self.cell_width, i * self.cell_height))

        for color, ghost in self.game.ghosts.items():
            ghost_img = self.ghosts[color]
            x, y = ghost.get_pos()
            surface.blit(ghost_img, (y * self.cell_width, x * self.cell_height))

        x, y = self.game.pacman.get_pos()
        rotate_angle = self.game.pacman.get_dir().angle()
        surface.blit(
            self.rotate_pacman(self.pacman[self.pacman_state], rotate_angle),
            (y * self.cell_width, x * self.cell_height),
        )

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
