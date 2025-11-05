from ...utils.colors import Color
from .game import FlappyGame
from ..base_frame_renderer import BaseFrameRenderer
from PIL import Image
from typing import Optional
import numpy as np


class FlappyFrameRenderer(BaseFrameRenderer[FlappyGame]):
    def __init__(self, game: FlappyGame):
        super().__init__(game)
        self.bird_color = Color.RED
        self.pipe_color = Color.GREEN

    def render(self, output_size: Optional[int] = None) -> Image.Image:
        if output_size is not None and output_size <= 0:
            raise ValueError(
                f"`output_size should be a positive integer, given: {output_size}"
            )
        if output_size is None:
            output_size = 256

        screenshot = np.full(
            (output_size, output_size, 3), fill_value=255, dtype=np.uint8
        )

        flappy_x = int(output_size * self.game.bird.x)
        flappy_y = int(output_size * self.game.bird.y)
        height = int(output_size * self.game.bird.size)

        screenshot[flappy_y : flappy_y + height, flappy_x : flappy_x + height] = (
            self.bird_color
        )

        for obstacle in self.game.obstacles:
            left = int(obstacle.x * output_size)
            right = int((obstacle.x + obstacle.width) * output_size)
            top = int(obstacle.top * output_size)
            bottom = int(obstacle.bottom * output_size)

            screenshot[:top, left:right] = self.pipe_color
            screenshot[bottom:, left:right] = self.pipe_color

        return Image.fromarray(screenshot)
