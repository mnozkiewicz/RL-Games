from pathlib import Path
from PIL import Image
from typing import Tuple
import numpy as np


def get_current_dir_path() -> Path:
    current_file_dir = Path(__file__).parent
    return current_file_dir


class CarTrack:
    def __init__(self) -> None:
        self._track_path = get_current_dir_path() / "racetrack.png"

        track_board = np.array(Image.open(self._track_path).convert("L"))
        track_board[track_board > 0] = 1
        track_board[track_board <= 0] = 0
        self._track_board = np.array(track_board, dtype=np.bool_)

    def check_car_collision(self, x: float, y: float) -> bool:
        y_pixel = int(y * self._track_board.shape[0])
        x_pixel = int(x * self._track_board.shape[1])
        return not self._track_board[y_pixel, x_pixel]

    def get_track_path(self) -> Path:
        return self._track_path

    def init_car_params(self) -> Tuple[float, float, int]:
        return (0.3, 0.5, 27)
