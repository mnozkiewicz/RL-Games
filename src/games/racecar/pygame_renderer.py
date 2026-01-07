import pygame
from ...utils.colors import Color
from .game import RacecarGame
from .car import Car
from ..base_pygame_renderer import BasePygameRenderer
from typing import Dict
from collections import defaultdict


class RacecarPygameRenderer(BasePygameRenderer[RacecarGame]):
    def __init__(self, game: RacecarGame):
        super().__init__(game)

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        self.screen_height = surface.get_height()
        self.screen_width = surface.get_width()

        # Track Surface
        track_surface = pygame.image.load(
            self.game.track.get_track_path()
        ).convert_alpha()
        self.track_surface = pygame.transform.scale(
            track_surface, (self.screen_width, self.screen_height)
        )

        # Car surface
        car_size = (
            self.screen_height * Car.CAR_LENGTH,
            self.screen_width * Car.CAR_WIDTH,
        )
        self.car_surface = pygame.Surface(car_size, pygame.SRCALPHA)
        pygame.draw.rect(self.car_surface, Color.RED, self.car_surface.get_rect())

    def _draw_car(self, surface: pygame.Surface) -> None:
        x_pos, y_pos = self.game.car.pos()
        car_center = self.screen_width * x_pos, self.screen_height * y_pos

        car_rotated = pygame.transform.rotate(self.car_surface, self.game.car.angle)

        rect = car_rotated.get_rect(center=car_center)
        surface.blit(car_rotated, rect)

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.track_surface, (0, 0))
        self._draw_car(surface)

    def get_key_map(self) -> Dict[int, int]:
        return defaultdict(
            lambda: -1,
            {
                pygame.K_UP: 0,
                pygame.K_LEFT: 1,
                pygame.K_RIGHT: 2,
                pygame.K_DOWN: 3,
                pygame.K_w: 0,
                pygame.K_a: 1,
                pygame.K_d: 2,
                pygame.K_s: 3,
            },
        )
