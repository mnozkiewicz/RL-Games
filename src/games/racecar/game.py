from ..base_game import BaseGame
from .car import Car
from .track import CarTrack
import numpy as np


class RacecarGame(BaseGame):
    def __init__(self, infinite: bool = True, is_ai_controlled: bool = False) -> None:
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled
        self.track = CarTrack()

        self._running: bool
        self._score: int
        self.car: Car
        self.reset()

    def reset(self) -> None:
        self._running = True
        self._score = 0
        self.car = Car(*self.track.init_car_params())

    def _check_collision(self) -> bool:
        x, y = self.car.pos()
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def end_game(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def step(self, action_label: int) -> float:
        self._score += 1
        self.car.move(action_label)

        if self._check_collision() or self.track.check_car_collision(*self.car.pos()):
            self.end_game()
            return -100

        return -1

    def is_running(self) -> bool:
        return self._running

    def processed_state(self) -> np.ndarray:
        return np.array([0], dtype=np.float32)

    def name(self) -> str:
        return "RACECAR"

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return self._score
