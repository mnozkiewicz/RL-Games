from .shapes import SHAPES, Shape
from collections import deque
import random


class ShapeGenerator:
    def __init__(self, buffer_size: int = 3) -> None:
        assert buffer_size >= 1

        self._current_shape = self._get_random_shape()
        self.future_shapes = deque(
            [self._get_random_shape() for _ in range(buffer_size)]
        )

    def _get_random_shape(self) -> Shape:
        return random.choice(SHAPES)(0, 4, random.randrange(4))

    def current_shape(self) -> Shape:
        return self._current_shape

    def next_shape(self) -> None:
        self._current_shape = self.future_shapes.popleft()
        self.future_shapes.append(self._get_random_shape())
