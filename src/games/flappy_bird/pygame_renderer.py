import pygame
from ...utils.colors import Color
from .game import FlappyGame
from ..base_pygame_renderer import BasePygameRenderer


class FlappyRenderer(BasePygameRenderer[FlappyGame]):
    def __init__(self, game: FlappyGame):
        super().__init__(game)

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        self.screen_width = surface.get_width()
        self.screen_height = surface.get_height()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(Color.WHITE)

        flappy_x = int(self.screen_width * self.game.bird.x)
        flappy_y = int(self.screen_height * self.game.bird.y)
        height = int(self.screen_height * self.game.bird.size)

        rect = pygame.Rect(flappy_x, flappy_y, height, height)
        pygame.draw.rect(surface, Color.RED, rect)

        for obstacle in self.game.obstacles:
            upper_rect = pygame.Rect(
                obstacle.x * self.screen_width,
                0,
                obstacle.width * self.screen_width,
                obstacle.top * self.screen_height,
            )

            lower_rect = pygame.Rect(
                obstacle.x * self.screen_width,
                obstacle.bottom * self.screen_height,
                obstacle.width * self.screen_width,
                (1 - obstacle.bottom) * self.screen_height,
            )

            pygame.draw.rect(surface, Color.GREEN, upper_rect)
            pygame.draw.rect(surface, Color.GREEN, lower_rect)
