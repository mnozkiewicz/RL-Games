import numpy as np
from typing import List
from random import randrange

from ..base_game import BaseGame
from .utils import Action
from .bird import Bird
from .obstacle import Obstacle

label_to_action = {
    -1: Action.NOTHING,
    0: Action.NOTHING,
    1: Action.JUMP,
}


class FlappyGame(BaseGame):
    OBSTACLE_TIME_SPAWN = 50
    MOVE_REWARD = 1
    DEATH_REWARD = 0

    def __init__(self, infinite: bool = True):
        self.bird: Bird = Bird()
        self.obstacles: List[Obstacle] = []
        self._score = 0
        self.last_spawn = 0

        self._running: bool = True

        self.infinite = infinite
        self.reset()

    def reset(self) -> None:
        self.bird.reset()
        self.obstacles = []
        self._score = 0
        self.last_spawn = 0
        self._running = True

    def death(self) -> int:
        if self.infinite:
            self.reset()
        else:
            self._running = False
        return FlappyGame.DEATH_REWARD

    def step(self, action_label: int) -> int:
        action = label_to_action[action_label]
        self.bird.step(action)

        for obstacle in self.obstacles:
            obstacle.step()
            if obstacle.collision(self.bird):
                return self.death()
            if obstacle.x + obstacle.width < self.bird.x:
                self._score += obstacle.pass_bird()

        if self.obstacles and self.obstacles[0].x < -0.1:
            self.obstacles.pop(0)

        if self.bird.y + self.bird.size >= 1.03 or self.bird.y <= -0.03:
            return self.death()

        self.last_spawn += 1
        if self.last_spawn > FlappyGame.OBSTACLE_TIME_SPAWN:
            self.obstacles.append(Obstacle())
            self.last_spawn = randrange(0, 20)

        return FlappyGame.MOVE_REWARD

    def is_running(self) -> bool:
        return self._running

    def processed_state(self) -> np.ndarray:  # TODO: implement state processing
        features: List[float] = []
        for obstacle in self.obstacles:
            if obstacle.x + obstacle.width > self.bird.x:
                features.append(obstacle.x + obstacle.width - self.bird.x)
                features.append(obstacle.top - self.bird.y)
                features.append(obstacle.bottom - self.bird.y)
                features.append(obstacle.width)
                break

        if not features:
            features.extend([1.0, -1.0, 1.0, 0.0])

        features.append(self.bird.y)
        features.append(1 - self.bird.y)
        features.append(self.bird.y_speed)
        features.append(self.bird.size)

        return np.array(features, dtype=np.float32)

    def name(self) -> str:
        return "Flappy Bird"

    @property
    def number_of_moves(self) -> int:
        return 2

    def score(self) -> int:
        return self._score
