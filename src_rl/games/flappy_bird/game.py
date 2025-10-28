import numpy as np
from ..base_game import BaseGame
from dataclasses import dataclass
from .utils import Action


@dataclass
class Obstacle:
    x: float
    speed: float = 0.01


label_to_action = {
    -1: Action.NOTHING,
    0: Action.NOTHING,
    1: Action.JUMP,
}


class FlappyGame(BaseGame):
    def __init__(self, infinite: bool = True):
        self.x: float = 0.3
        self.y: float = 0.5

        self.g = 1.0  # Acceleraration
        self.t = 0.1
        self.y_speed = 0.0
        self.flap_strength = -0.3
        self.size = 0.1

        self._running: bool = True

        self.reset()

    def reset(self):
        self.x = 0.3
        self.y = 0.5

        self._running = True

    def step(self, action_label: int) -> int:
        action = label_to_action[action_label]
        print(action)
        if action == Action.NOTHING:
            self.y_speed += self.g * self.t
        else:
            self.y_speed = self.flap_strength

        self.y += self.y_speed * self.t

        print(self.y_speed)

        if self.y >= 1.0 or self.y <= 0:
            self._running = False

        return 1

    def is_running(self) -> bool:
        return self._running

    def processed_state(self) -> np.ndarray:
        return np.zeros((1, 1))

    def name(self) -> str:
        return "Flappy Bird"

    @property
    def number_of_moves(self) -> int:
        return 2


#     def reset(self):

#     def step(self, action_label: int) -> int:

#     def is_running(self) -> bool:

#     def processed_state(self) -> np.ndarray:
