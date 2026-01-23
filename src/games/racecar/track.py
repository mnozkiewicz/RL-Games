from pathlib import Path
from PIL import Image
from typing import Tuple
import numpy as np
import json
from enum import Enum


def get_current_dir_path() -> Path:
    current_file_dir = Path(__file__).parent
    return current_file_dir


class Event(Enum):
    NOTHING = 1
    OUT_OF_TRACK = 2
    NEXT_CHECKPOINT = 3
    WRONG_CHECHKPOINT = 4
    FULL_CIRCLE = 5


class CarTrack:
    def __init__(self) -> None:
        self._track_path = get_current_dir_path() / "tracks/racetrack2.png"
        self._mid_points_path = get_current_dir_path() / "tracks/checkpoints2.json"

        self._track_board = self._read_board()
        self._checkpoints, self.angles = self._read_checkpoints()
        self._mid_point_lengths = np.linalg.norm(self._checkpoints, axis=1) ** 2

        # Car progress tracking
        self._last_checkpoint = -1
        self._next_checkpoint = 0

    def _read_board(self) -> np.ndarray:
        track_board = np.array(Image.open(self._track_path).convert("RGB"))

        red = track_board[:, :, 0] == 255
        green = track_board[:, :, 1] == 255
        blue = track_board[:, :, 2] == 255

        greyscale = np.zeros_like(track_board[:, :, 0], dtype=np.int32)
        greyscale[red] = 1
        greyscale[green] = 2
        greyscale[blue] = 3

        return greyscale

    def _read_checkpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        with open(self._mid_points_path, "r") as f:
            checkpoint_json = json.load(f)
            mid_points: list[list[int]] = checkpoint_json["checkpoints"]
            angles: list[int] = checkpoint_json["angles"]

        starting_checkpoint = np.random.randint(0, len(mid_points))
        # starting_checkpoint = 0

        for _ in range(starting_checkpoint):
            mid_points.append(mid_points.pop(0))
            angles.append(angles.pop(0))

        return np.array(mid_points), np.array(angles)

    def _which_checkpoint(self, x: int, y: int) -> int:
        point = np.array([x, y])
        return np.argmin(
            -2 * (self._checkpoints @ point.T) + self._mid_point_lengths
        ).item()

    def distance_to_checkpoint(self, x: float, y: float) -> float:
        checkpoint_x, checkpoint_y = self._checkpoints[self._next_checkpoint]

        checkpoint_x /= self._track_board.shape[1]
        checkpoint_y /= self._track_board.shape[0]
        return float((x - checkpoint_x) ** 2 + (y - checkpoint_y) ** 2)

    def check_car_collision(self, x: float, y: float) -> Event:
        y_pixel = int(y * self._track_board.shape[0])
        x_pixel = int(x * self._track_board.shape[1])

        match self._track_board[y_pixel, x_pixel].item():
            case 0:
                return Event.OUT_OF_TRACK
            case 1 | 3:
                cur_checkpoint = self._which_checkpoint(x_pixel, y_pixel)
                if cur_checkpoint == self._last_checkpoint:
                    return Event.NOTHING

                if cur_checkpoint != self._next_checkpoint:
                    self._last_checkpoint = cur_checkpoint
                    return Event.WRONG_CHECHKPOINT

                if self._next_checkpoint == 0 and self._last_checkpoint >= 0:
                    # print("FULL circle done")
                    return Event.FULL_CIRCLE

                self._next_checkpoint = (self._next_checkpoint + 1) % len(
                    self._checkpoints
                )
                # print(self._last_checkpoint, cur_checkpoint, self._next_checkpoint)
                self._last_checkpoint = cur_checkpoint
                return Event.NEXT_CHECKPOINT

            case _:
                return Event.NOTHING

    def ray_cast(
        self,
        x: float,
        y: float,
        angle: int,
        num_rays: int = 40,
    ) -> np.ndarray:
        h, w = self._track_board.shape
        y0 = int(y * h)
        x0 = int(x * w)

        max_distance = max(h, w)

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        distances = np.zeros(num_rays, dtype=np.float32)

        shift = angle // (360 // num_rays)

        for i, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)

            angle_pos = (i + shift) % num_rays

            for d in range(1, max_distance):
                xi = int(x0 + dx * d)
                yi = int(y0 + dy * d)

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    distances[angle_pos] = d / max_distance
                    break

                if self._track_board[yi, xi] == 0:
                    distances[angle_pos] = d / max_distance
                    break
                # if self._track_board[yi, xi] == 1 or self._track_board[yi, xi] == 3:
                #     distances[angle_pos] = d
            else:
                distances[angle_pos] = 1.0

        return distances

    def get_track_path(self) -> Path:
        return self._track_path

    def init_car_params(self) -> Tuple[float, float, int]:
        x_start, y_start = self._checkpoints[0]
        x_start /= self._track_board.shape[0]
        y_start /= self._track_board.shape[1]
        return (x_start, y_start, self.angles[0])
