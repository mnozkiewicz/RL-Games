import numpy as np
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Any


def check_output_shape(f: Callable[[Any], np.ndarray]) -> Callable[[Any], np.ndarray]:
    @wraps(f)
    def check_array(self: Any) -> np.ndarray:
        out = f(self)
        assert out.shape == (4, 4)
        return out

    return check_array


class Shape(ABC):
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y
        self.rotation: int = 0

    @abstractmethod
    @check_output_shape
    def mask(self) -> np.ndarray: ...

    def move_shape(self) -> None:
        self.x += 1
        self.y += 1


class LShape(Shape):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)

    @check_output_shape
    def mask(self) -> np.ndarray:
        match self.rotation:
            case 0:
                return np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ]
                )
            case 1:
                return np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 1],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                    ]
                )
            case 2:
                return np.array(
                    [
                        [0, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ]
                )
            case 3:
                return np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 1, 1, 1],
                        [0, 0, 0, 0],
                    ]
                )
            case _:
                raise ValueError("Undefined rotation")
