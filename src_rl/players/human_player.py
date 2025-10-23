import numpy as np
import pygame
from src_rl.games.base_game import BaseGame
from .base_player import BasePlayer
from pygame.event import Event


def key_to_action_map(event_type: Event) -> int:
    match event_type:
        case pygame.K_UP | pygame.K_w:
            return 0
        case pygame.K_DOWN | pygame.K_s:
            return 1
        case pygame.K_LEFT | pygame.K_a:
            return 2
        case pygame.K_RIGHT | pygame.K_d:
            return 3
        case _:
            return -1


def map_events(events: list[pygame.event.Event]) -> list[int]:
    actions: list[int] = []
    for event in events:
        if event.type == pygame.KEYDOWN:
            action = key_to_action_map(event.key)
            actions.append(action)
    return actions


class HumanPlayer(BasePlayer):
    def __init__(self, game: BaseGame):
        super().__init__(game)
        self.last_action = -1

    def handle_events(self, events: list[Event]) -> None:
        actions = map_events(events)
        if len(actions) > 0:
            self.last_action = actions[-1]

    def move(self, state: np.ndarray) -> int:
        return self.last_action

    def learn(
        self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool
    ) -> None:
        pass
