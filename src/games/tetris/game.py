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
    STEP_REWARD = 0.1
    DEATH_REWARD = -500.0
    ROW_REMOVED_REWARD = 100.0
    BLOCK_PUT_REMOVED = 10

    def __init__(self, infinite: bool = True, is_ai_controlled: bool = False) -> None:
        self.infinite = infinite
        self.is_ai_controlled = is_ai_controlled

        self.board: np.ndarray
        self.shape_manager: ShapeGenerator
        self.move_counter: int

        self._running: bool = True
        self.reset()

    def reset(self) -> None:
        self._score = 0
        self._running = True
        self.board = np.zeros((20, 10), dtype=np.int32)
        self.move_counter = 0
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

    def _end_game(self) -> None:
        if not self.infinite:
            self._running = False
        else:
            self.reset()

    def _remove_rows(self) -> int:
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

        block_put_reward = 0
        if (
            collision == CollisionType.LOWER_WALL
            or collision == CollisionType.OTHER_SHAPE_FROM_UP
        ):
            if shape.move_counter <= 1:
                self._end_game()
                return TetrisGame.DEATH_REWARD

            self._add_shape(shape)
            self.shape_manager.next_shape()
            block_put_reward = 1

        rows_removed = self._remove_rows()
        self._score += rows_removed

        self.move_counter += 1
        if self.move_counter % 4 == 0:
            self.step(-1)

        return (
            rows_removed * TetrisGame.ROW_REMOVED_REWARD
            + TetrisGame.STEP_REWARD
            + block_put_reward * TetrisGame.BLOCK_PUT_REMOVED
        )

    def is_running(self) -> bool:
        return self._running

    def _processed_state(self) -> np.ndarray:
        current_board = np.zeros_like(self.board)
        current_board[self.board > 0] = 1
        current_board_vector = current_board.flatten()

        cur_shape = self.shape_manager.current_shape()
        cur_shape_mask = cur_shape.mask().flatten()
        cur_shape_mask = np.pad(
            cur_shape_mask, (0, 6 - cur_shape_mask.shape[0]), constant_values=0
        )

        position = [cur_shape.x, cur_shape.y]

        rotation = [0, 0, 0, 0]
        rotation[cur_shape.rotation] = 1

        shape_id = [0] * 7
        shape_id[cur_shape.shape_id] = 1

        whole_state = np.concatenate(
            (current_board_vector, cur_shape_mask, position, rotation, shape_id),
            dtype=np.float32,
        )

        return whole_state

    def _raw_state(self) -> np.ndarray:
        raise NotImplementedError(
            "Raw pixel state vesrsion not available for tetris for now"
        )

    def name(self) -> str:
        return "Tetris"

    @property
    def number_of_moves(self) -> int:
        return 4

    def score(self) -> int:
        return self._score
