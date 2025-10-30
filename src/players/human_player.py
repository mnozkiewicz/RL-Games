import numpy as np
import pygame
from ..games.base_game import BaseGame
from .base_player import BasePlayer
from pygame.event import Event
from typing import Dict, List


class HumanPlayer(BasePlayer):
    """
    A 'Player' that is controlled by a human using the keyboard.
    Every game should have key maps defined.
    """

    def __init__(self, game: BaseGame, key_map: Dict[int, int]):
        super().__init__(game)
        # We initialize last_action to -1, every game should ignore -1
        self.last_action = -1
        self.key_map = key_map

    def map_events(self, events: List[pygame.event.Event]) -> List[int]:
        """
        Filters a list of pygame events and returns all mapped actions.
        """
        actions: List[int] = []
        for event in events:
            if event.type == pygame.KEYDOWN:
                action = self.key_map[event.key]
                if action != -1:  # Only add valid, mapped actions
                    actions.append(action)
        return actions

    def handle_events(self, events: List[Event]) -> None:
        """
        Scans pygame events for key presses and updates the last action.
        """
        actions = self.map_events(events)
        if len(actions) > 0:
            # If multiple keys were pressed, take the last one.
            self.last_action = actions[-1]
        else:
            self.last_action = -1

    def move(self, state: np.ndarray) -> int:
        """
        Returns the last action that was captured by `handle_events`.
        """
        return self.last_action

    def feedback(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        pass
