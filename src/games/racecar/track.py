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
        self._checkpoints = self._read_mid_points()
        self._mid_point_lengths = np.linalg.norm(self._checkpoints, axis=1) ** 2
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

    def _read_mid_points(self) -> np.ndarray:
        with open(self._mid_points_path, "r") as f:
            mid_points = json.load(f)
        return np.array(mid_points)

    def _which_checkpoint(self, x: int, y: int) -> int:
        point = np.array([x, y])
        return np.argmin(
            -2 * (self._checkpoints @ point.T) + self._mid_point_lengths
        ).item()

    def check_car_collision(self, x: float, y: float) -> Event:
        y_pixel = int(y * self._track_board.shape[0])
        x_pixel = int(x * self._track_board.shape[1])

        match self._track_board[y_pixel, x_pixel].item():
            case 0:
                return Event.OUT_OF_TRACK
            case 1 | 3:
                checkpoint_number = self._which_checkpoint(x_pixel, y_pixel)
                if checkpoint_number == self._next_checkpoint:
                    self._next_checkpoint += 1
                    return Event.NEXT_CHECKPOINT

                if (
                    self._next_checkpoint == len(self._checkpoints)
                    and checkpoint_number == 0
                ):
                    return Event.FULL_CIRCLE

                return Event.WRONG_CHECHKPOINT
            case _:
                return Event.NOTHING

    def ray_cast(
        self,
        x: float,
        y: float,
        num_rays: int = 16,
    ) -> np.ndarray:
        h, w = self._track_board.shape
        y0 = int(y * h)
        x0 = int(x * w)

        max_distance = max(h, w)

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        distances = np.zeros(num_rays, dtype=np.float32)

        for i, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)

            for d in range(1, max_distance):
                xi = int(x0 + dx * d)
                yi = int(y0 + dy * d)

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    distances[i] = d
                    break

                if self._track_board[yi, xi] == 0:
                    distances[i] = d
                    break
            else:
                distances[i] = max_distance

        return distances

    def get_track_path(self) -> Path:
        return self._track_path

    def init_car_params(self) -> Tuple[float, float, int]:
        return (0.2, 0.25, 27)
