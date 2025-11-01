from ...utils.colors import Color
from .game import SnakeGame
from ..base_frame_renderer import BaseFrameRenderer
from PIL import Image
from typing import Optional
import numpy as np


class SnakeFrameRenderer(BaseFrameRenderer[SnakeGame]):
    def __init__(self, game: SnakeGame):
        super().__init__(game)
        self.head_color = Color.RED
        self.tail_color = Color.GREEN
        self.food_color = Color.BLUE

    def render(self, output_size: Optional[int] = None) -> Image.Image:
        if output_size is not None and output_size <= 0:
            raise ValueError(
                f"`output_size should be a positive integer, given: {output_size}"
            )

        state = self.game.get_state()
        size = state.board_size
        screenshot = np.full((size, size, 3), fill_value=255, dtype=np.uint8)

        for tail_pos in state.tail:
            x, y = tail_pos
            screenshot[x, y] = Color.GREEN

        food_x, food_y = state.food
        screenshot[food_x, food_y] = Color.BLUE

        head_x, head_y = state.head
        screenshot[head_x, head_y] = Color.RED

        img = Image.fromarray(screenshot)
        if output_size is None:
            return img

        img = img.resize((output_size, output_size), resample=Image.Resampling.NEAREST)
        return img
