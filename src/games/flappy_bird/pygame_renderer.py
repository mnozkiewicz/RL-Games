import pygame
from .game import FlappyGame
from ..base_pygame_renderer import BasePygameRenderer
from pathlib import Path
from typing import Optional, Tuple, Dict


class FlappyRenderer(BasePygameRenderer[FlappyGame]):
    def __init__(self, game: FlappyGame):
        super().__init__(game)

    def resize(self, surface: pygame.Surface, size: Tuple[int, int]) -> pygame.Surface:
        return pygame.transform.scale(surface, size).convert_alpha()

    def load_image(
        self, path: Path, size: Optional[Tuple[int, int]] = None
    ) -> pygame.Surface:
        image = pygame.image.load(path).convert_alpha()
        if size is None:
            return image
        return self.resize(image, size)

    def load_assets(self) -> None:
        # Path to folder containing assets
        asset_path = Path("assets/flappy")

        self.background = self.load_image(
            asset_path / "background.png", (self.screen_width, self.screen_height)
        )
        self.pipe = self.load_image(asset_path / "pipe.png")

        self.bird_size = int(self.game.bird.size * self.screen_height)

        self.upflap = self.load_image(
            asset_path / "upflap.png", (self.bird_size, self.bird_size)
        )
        self.downflap = self.load_image(
            asset_path / "downflap.png", (self.bird_size, self.bird_size)
        )
        self.midflap = self.load_image(
            asset_path / "midflap.png", (self.bird_size, self.bird_size)
        )

        self.state_to_image: Dict[int, pygame.Surface] = {
            0: self.upflap,
            1: self.downflap,
            2: self.midflap,
        }

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        self.screen_width = surface.get_width()
        self.screen_height = surface.get_height()

        self.obstacles: Dict[int, Tuple[pygame.Surface, pygame.Surface]] = {}
        self.bird_state = 2

        self.load_assets()

    def update_bird_state(self) -> None:
        self.bird_state = (self.bird_state + 1) % 3

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.background, (0, 0))
        self.update_bird_state()

        flappy_x = int(self.screen_width * self.game.bird.x)
        flappy_y = int(self.screen_height * self.game.bird.y)
        surface.blit(self.state_to_image[self.bird_state], (flappy_x, flappy_y))

        seen_obstacles = {
            obstacle.get_id(): obstacle for obstacle in self.game.obstacles
        }

        for key in list(self.obstacles.keys()):
            if key not in seen_obstacles:
                self.obstacles.pop(key)

        for key, obstacle in seen_obstacles.items():
            if key not in self.obstacles:
                self.obstacles[key] = (
                    self.resize(
                        self.pipe,
                        (
                            int(obstacle.width * self.screen_width),
                            int(obstacle.top * self.screen_height),
                        ),
                    ),
                    self.resize(
                        self.pipe,
                        (
                            int(obstacle.width * self.screen_width),
                            int((1 - obstacle.bottom) * self.screen_height),
                        ),
                    ),
                )

        for key, obstacle in seen_obstacles.items():
            pipe_uppper, pipe_lower = self.obstacles[key]
            surface.blit(pipe_uppper, (int(obstacle.x * self.screen_width), 0))
            surface.blit(
                pipe_lower,
                (
                    int(obstacle.x * self.screen_width),
                    int(obstacle.bottom * self.screen_height),
                ),
            )
