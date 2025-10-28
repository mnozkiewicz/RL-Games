import pygame
from .gui_config import GUIConfig
from ..games.base_game import BaseGame, GameType
from ..games.base_renderer import BaseRenderer
from ..players.base_player import BasePlayer
from ..players.event_handler import IEventHandler
from typing import List


class GameGui:
    """
    Manages the main game window, event loop, and visualization.
    """

    def __init__(
        self,
        renderer: BaseRenderer[GameType],
        game: BaseGame,
        config: GUIConfig,
        player: BasePlayer,
    ):
        """
        Args:
            renderer: An instance of BaseRenderer subclass responsible
                for drawing the game.
            game: An instance of BaseGame subclass containing the
                game's logic.
            config: A GUIConfig object with settings like window size
                and frame rate.
            player: An instance of BasePlayer subclass that will
                decide on actions.
        """
        # Injected Dependencies
        self.renderer = renderer
        self.config = config
        self.game = game
        self.player = player

        # Pygame State Variables
        self.screen: pygame.Surface  # The main display surface
        self.clock: pygame.time.Clock  # Manages the frame rate
        self.running: bool  # Controls the main game loop

    def check_if_quit(self, events: List[pygame.event.Event]) -> None:
        """
        Scans a list of events and sets self.running to False if
        a QUIT event (like closing the window) is found.
        """
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

    def gather_events(self) -> List[pygame.event.Event]:
        """
        Gets all pending pygame events and checks for a quit event.
        """
        events = list(pygame.event.get())
        self.check_if_quit(events)
        return events

    def run_game(self) -> None:
        """
        Initializes pygame and runs the main game loop.

        This loop handles events, gets actions from the player,
        steps the game engine, provides feedback to the player,
        and renders the game.
        """
        pygame.init()

        self.running = True
        self.screen = pygame.display.set_mode(
            (self.config.pixel_height, self.config.pixel_height)
        )
        pygame.display.set_caption(self.game.name())
        self.clock = pygame.time.Clock()

        state = self.game.processed_state()

        while self.running:
            #  Handle user input
            events = self.gather_events()
            # Pass events to the player (if it's a human/event handler)
            if isinstance(self.player, IEventHandler):
                self.player.handle_events(events)

            action = self.player.move(state)
            reward = self.game.step(action)
            new_state = self.game.processed_state()
            is_terminal = not self.game.is_running()

            self.player.feedback(state, action, reward, new_state, is_terminal)
            state = new_state

            self.renderer.draw(self.screen)
            pygame.display.flip()

            self.clock.tick(self.config.frame_rate)
            self.running = self.running and not is_terminal

        pygame.quit()
