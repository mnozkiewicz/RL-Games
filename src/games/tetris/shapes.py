import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Tuple, Type


class Shape(ABC):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        self.x: int = x
        self.y: int = y

        assert 0 <= rotation < 4, "rotation should be one of [0, 1, 2, 3]"

        self.rotation: int = rotation
        self.move_counter = 0

    @abstractmethod
    def mask(self) -> np.ndarray: ...

    def move(self, action: int = -1) -> None:
        self.move_counter += 1
        if action == -1:
            self.x += 1
        elif action == 0:
            self.rotation = (self.rotation + 1) % 4
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.y += 1
        else:
            raise ValueError(f"Received wrong action type: {action}")

    def rollback(self, action: int = -1) -> None:
        self.move_counter -= 1

        if action == -1:
            self.x -= 1
        elif action == 0:
            self.rotation -= 1
            if self.rotation == -1:
                self.rotation = 3
        elif action == 1:
            self.x -= 1
        elif action == 2:
            self.y += 1
        elif action == 3:
            self.y -= 1
        else:
            raise ValueError(f"Received wrong action type: {action}")


class LShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0:
                return np.array(
                    [
                        [1, 0],
                        [1, 0],
                        [1, 1],
                    ],
                    dtype=np.int32,
                )
            case 1:
                return np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 0],
                    ],
                    dtype=np.int32,
                )
            case 2:
                return np.array(
                    [
                        [1, 1],
                        [0, 1],
                        [0, 1],
                    ],
                    dtype=np.int32,
                )
            case 3:
                return np.array(
                    [
                        [0, 0, 1],
                        [1, 1, 1],
                    ],
                    dtype=np.int32,
                )
            case _:
                raise ValueError("Undefined rotation")


class JShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0:
                return np.array(
                    [
                        [0, 1],
                        [0, 1],
                        [1, 1],
                    ],
                    dtype=np.int32,
                )
            case 1:
                return np.array(
                    [
                        [1, 0, 0],
                        [1, 1, 1],
                    ],
                    dtype=np.int32,
                )
            case 2:
                return np.array(
                    [
                        [1, 1],
                        [1, 0],
                        [1, 0],
                    ],
                    dtype=np.int32,
                )
            case 3:
                return np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                    ],
                    dtype=np.int32,
                )
            case _:
                raise ValueError("Undefined rotation")


class TShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0:
                return np.array(
                    [
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                    dtype=np.int32,
                )
            case 1:
                return np.array([[0, 1], [1, 1], [0, 1]], dtype=np.int32)
            case 2:
                return np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                    ],
                    dtype=np.int32,
                )
            case 3:
                return np.array([[1, 0], [1, 1], [1, 0]], dtype=np.int32)
            case _:
                raise ValueError("Undefined rotation")


class ZShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0 | 2:
                return np.array(
                    [
                        [1, 1, 0],
                        [0, 1, 1],
                    ],
                    dtype=np.int32,
                )
            case 1 | 3:
                return np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int32)
            case _:
                raise ValueError("Undefined rotation")


class SShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0 | 2:
                return np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 0],
                    ],
                    dtype=np.int32,
                )
            case 1 | 3:
                return np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
            case _:
                raise ValueError("Undefined rotation")


class IShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0 | 2:
                return np.array(
                    [
                        [1],
                        [1],
                        [1],
                        [1],
                    ],
                    dtype=np.int32,
                )
            case 1 | 3:
                return np.array(
                    [
                        [1, 1, 1, 1],
                    ],
                    dtype=np.int32,
                )
            case _:
                raise ValueError("Undefined rotation")


class OShape(Shape):
    def __init__(self, x: int, y: int, rotation: int) -> None:
        super().__init__(x, y, rotation)

    def mask(self) -> npt.NDArray[np.int32]:
        match self.rotation:
            case 0 | 1 | 2 | 3:
                return np.array([[1, 1], [1, 1]], dtype=np.int32)
            case _:
                raise ValueError("Undefined rotation")


SHAPES: Tuple[Type[Shape], ...] = (
    LShape,
    JShape,
    TShape,
    ZShape,
    SShape,
    IShape,
    OShape,
)
