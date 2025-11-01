from typing import Tuple
from ..games.base_game import BaseGame
from .base_pygame_renderer import BasePygameRenderer

from .snake.game import SnakeGame
from .snake.pygame_renderer import SnakePygameRenderer
from .flappy_bird.game import FlappyGame
from .flappy_bird.pygame_renderer import FlappyRenderer


def create_game_and_renderer(
    game_name: str, infinite: bool
) -> Tuple[BaseGame, BasePygameRenderer[BaseGame]]:
    """
    Factory function to initialize the correct game and renderer.
    """
    if game_name == "snake":
        snake_game = SnakeGame(board_size=15, infinite=infinite)
        snake_renderer = SnakePygameRenderer(snake_game)
        return snake_game, snake_renderer
    elif game_name == "flappy":
        flappy_game = FlappyGame(infinite=infinite)
        flappy_renderer = FlappyRenderer(flappy_game)
        return flappy_game, flappy_renderer
    else:
        raise ValueError(f"Unknown game: {game_name}")
