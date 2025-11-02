from random import uniform
from .bird import Bird


def segment_col(left1: float, right1: float, left2: float, right2: float) -> bool:
    return (left1 <= left2 <= right1) or (left2 <= left1 <= right2)


class Obstacle:
    OBSTACLE_SPEED = 0.02

    def __init__(self, obstacle_id: int) -> None:
        self._obstacle_id = obstacle_id
        self.x = 1.0
        self.width = 0.08
        self.hole = uniform(0.2, 0.4)
        self.top = uniform(0, 1 - self.hole - 0.1)
        self.bottom = self.top + self.hole

        self.passed_bird = False

    def step(self) -> None:
        self.x -= Obstacle.OBSTACLE_SPEED

    def upper_collision(self, bird: Bird) -> bool:
        return segment_col(
            self.x, self.x + self.width, bird.x, bird.x + bird.size
        ) and segment_col(0, self.top, bird.y, bird.y + bird.size)

    def lower_collision(self, bird: Bird) -> bool:
        return segment_col(
            self.x, self.x + self.width, bird.x, bird.x + bird.size
        ) and segment_col(self.bottom, 1.0, bird.y, bird.y + bird.size)

    def collision(self, bird: Bird) -> bool:
        return self.upper_collision(bird) or self.lower_collision(bird)

    def pass_bird(self) -> int:
        if not self.passed_bird:
            self.passed_bird = True
            return 1
        else:
            return 0

    def get_id(self) -> int:
        return self._obstacle_id
