from ..base_game import BaseGame
from .car import Car
from .track import CarTrack, Event
import numpy as np
from typing import Literal


class RacecarGame(BaseGame):
    STEP_REWARD: float = -1.0
    STANDING_STILL_PENALTY: float = -1.0
    CHECKPOINT_DISTANCE_CHANGE: float = 10.0
    NEXT_CHECKPOINT_REWARD: float = 10.0
    WRONG_CHECKPOINT_REWARD: float = -10.0
    DEATH_REWARD: float = -40.0
    WIN_REWARD: float = 100.0

    def __init__(
        self,
        state_type: Literal["processed_state", "raw_pixels"],
        infinite: bool = True,
        is_ai_controlled: bool = False,
    ) -> None:
        super().__init__(state_type=state_type)
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled
        self.track: CarTrack

        self._running: bool
        self._score: int
        self.car: Car
        self.last_dist: float
        self.reset()

    def reset(self) -> None:
        self._running = True
        self._score = 0
        self.track = CarTrack()
        self.car = Car(*self.track.init_car_params())
        self.last_dist = 0.0

    def _check_collision(self) -> bool:
        x, y = self.car.pos()
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def end_game(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def step(self, action_label: int) -> float:
        if action_label == 4:
            action_label = -1

        self.car.move(action_label)
        event = self.track.check_car_collision(*self.car.pos())

        if self._check_collision():
            self.end_game()
            return RacecarGame.DEATH_REWARD

        if self.car.speed <= 1e-5:
            return RacecarGame.STANDING_STILL_PENALTY

        match event:
            case Event.NOTHING:
                dist = self.track.distance_to_checkpoint(*self.car.pos())
                # dist_change = self.last_dist - dist
                self.last_dist = dist
                return (
                    RacecarGame.STEP_REWARD * (Car.MAX_SPEED - self.car.speed)
                    # RacecarGame.CHECKPOINT_DISTANCE_CHANGE * dist_change
                )
            case Event.OUT_OF_TRACK:
                self.end_game()
                return RacecarGame.DEATH_REWARD
            case Event.NEXT_CHECKPOINT:
                self._score += 1
                return RacecarGame.NEXT_CHECKPOINT_REWARD
            case Event.WRONG_CHECHKPOINT:
                return RacecarGame.WRONG_CHECKPOINT_REWARD
            case Event.FULL_CIRCLE:
                self.end_game()
                return RacecarGame.WIN_REWARD

    def is_running(self) -> bool:
        return self._running

    def _processed_state(self) -> np.ndarray:
        pos = np.array([self.car.speed])
        direction = np.array(self.car.vector)
        ray_cast = self.track.ray_cast(*self.car.pos(), self.car.angle, num_rays=40)
        state = np.concatenate(
            (
                pos,
                direction,
                ray_cast,
            ),
            dtype=np.float32,
        )

        return state

    def _raw_state(self) -> np.ndarray:
        raise NotImplementedError(
            "Raw pixel state vesrsion not available for racecar for now"
        )

    def name(self) -> str:
        return "RACECAR"

    @property
    def number_of_moves(self) -> int:
        return 5

    def score(self) -> int:
        return self._score
