from ..base_game import BaseGame
from .shape_generator import ShapeGenerator
from .shapes import Shape
from enum import Enum
import numpy as np


class CollisionType(Enum):
    VERTICAL_WALL = 1
    LOWER_WALL = 2
    OTHER_SHAPE_FROM_SIDE = 3
    OTHER_SHAPE_FROM_UP = 4
    NO_COLLISION = 5


class TetrisGame(BaseGame):
    def __init__(self, infinite: bool = True, is_ai_controlled: bool = False) -> None:
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled

        self.board: np.ndarray
        self.shape_manager: ShapeGenerator

        self._running: bool = True
        self.reset()

    def reset(self) -> None:
        self._score = 0
        self._running = True
        self.board = np.zeros((20, 10), dtype=np.int32)
        self.shape_manager = ShapeGenerator()

    def _detect_collision(self, shape: Shape, action_label: int) -> CollisionType:
        mask = shape.mask()
        x0, x1 = shape.x, shape.x + mask.shape[0]
        y0, y1 = shape.y, shape.y + mask.shape[1]

        if x0 < 0 or x1 > 20:
            return CollisionType.LOWER_WALL

        if y0 < 0 or y1 > 10:
            return CollisionType.VERTICAL_WALL

        slice = self.board[x0:x1, y0:y1]
        collision_count = np.sum((slice * mask))

        if (collision_count > 0).item():
            return (
                CollisionType.OTHER_SHAPE_FROM_SIDE
                if action_label in [2, 3]
                else CollisionType.OTHER_SHAPE_FROM_UP
            )

        return CollisionType.NO_COLLISION

    def _add_shape(self, shape: Shape) -> None:
        mask = shape.mask()
        x0, x1 = shape.x, shape.x + mask.shape[0]
        y0, y1 = shape.y, shape.y + mask.shape[1]
        self.board[x0:x1, y0:y1] += mask

    def end_game(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def remove_rows(self) -> int:
        last_row, column_count = self.board.shape
        count = 0

        for i in range(last_row - 1, -1, -1):
            if np.all(self.board[i] > 0):
                count += 1
                self.board[i, :] = 0
            else:
                if count <= 0:
                    continue
                for j in range(column_count):
                    if self.board[i, j] > 0:
                        self.board[i, j] = 0
                        self.board[i + count, j] = 1

        return count

    def step(self, action_label: int) -> float:
        shape = self.shape_manager.current_shape()
        shape.move(action=action_label)
        collision = self._detect_collision(shape, action_label)

        if collision != CollisionType.NO_COLLISION:
            shape.rollback(action=action_label)

        if (
            collision == CollisionType.LOWER_WALL
            or collision == CollisionType.OTHER_SHAPE_FROM_UP
        ):
            if shape.move_counter <= 1:
                self.end_game()
                return -1

            self._add_shape(shape)
            self.shape_manager.next_shape()

        rows_removed = self.remove_rows()
        self._score += rows_removed

        if shape.move_counter % 4 == 0:
            self.step(-1)

        return rows_removed

    def is_running(self) -> bool:
        return self._running

    def processed_state(self) -> np.ndarray:
        return np.array([[0]], dtype=np.float32)

    def name(self) -> str:
        return "Tetris"

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return self._score
