import pygame
from .gui_config import GUIConfig
from ..games.base_game import BaseGame, GameType
from ..games.base_renderer import BaseRenderer


class GameGui:
    def __init__(
        self, renderer: BaseRenderer[GameType], game: BaseGame, config: GUIConfig
    ):
        self.renderer = renderer
        self.config = config
        self.game = game

        self.screen: pygame.Surface
        self.clock: pygame.time.Clock
        self.running: bool

    def check_if_quit(self, events: list[pygame.event.Event]):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

    def gather_events(self) -> list[pygame.event.Event]:
        events = list(pygame.event.get())
        self.check_if_quit(events)
        return events

    def run_game(self) -> None:
        pygame.init()

        self.running = True
        self.screen = pygame.display.set_mode(
            (self.config.pixel_height, self.config.pixel_height)
        )

        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        while self.running:
            events = self.gather_events()
            actions = self.renderer.handle_events(events)

            _ = self.game.step(actions)
            self.renderer.draw(self.screen)

            pygame.display.flip()

            self.clock.tick(self.config.frame_rate)
            self.running = self.running and self.game.is_running()

        pygame.quit()
