from ..games.base_game import BaseGame

from .snake.game import SnakeGame
from .flappy_bird.game import FlappyGame


def create_game(game_name: str, infinite: bool) -> BaseGame:
    """
    Factory function to initialize the correct game.
    """
    if game_name == "snake":
        snake_game = SnakeGame(board_size=15, infinite=infinite)
        return snake_game
    elif game_name == "flappy":
        flappy_game = FlappyGame(infinite=infinite)
        return flappy_game
    else:
        raise ValueError(f"Unknown game: {game_name}")
