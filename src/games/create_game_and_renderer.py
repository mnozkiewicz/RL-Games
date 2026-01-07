from typing import Tuple
from ..games.base_game import BaseGame
from .base_pygame_renderer import BasePygameRenderer

from .snake.game import SnakeGame
from .snake.pygame_renderer import SnakePygameRenderer
from .flappy_bird.game import FlappyGame
from .flappy_bird.pygame_renderer import FlappyPygameRenderer
from .pacman.game import PacmanGame
from .pacman.pygame_renderer import PacmanPygameRenderer
from .tetris.game import TetrisGame
from .tetris.pygame_renderer import TetrisPygameRenderer
from .racecar.game import RacecarGame
from .racecar.pygame_renderer import RacecarPygameRenderer


def create_game_and_renderer(
    game_name: str, infinite: bool, is_ai_controlled: bool
) -> Tuple[BaseGame, BasePygameRenderer[BaseGame]]:
    """
    Factory function to initialize the correct game and renderer.
    """
    if game_name == "snake":
        snake_game = SnakeGame(
            board_size=15, infinite=infinite, is_ai_controlled=is_ai_controlled
        )
        snake_renderer = SnakePygameRenderer(snake_game)
        return snake_game, snake_renderer
    elif game_name == "flappy":
        flappy_game = FlappyGame(infinite=infinite, is_ai_controlled=is_ai_controlled)
        flappy_renderer = FlappyPygameRenderer(flappy_game)
        return flappy_game, flappy_renderer
    elif game_name == "pacman":
        pacman_game = PacmanGame(infinite=infinite, is_ai_controlled=is_ai_controlled)
        pacman_renderer = PacmanPygameRenderer(pacman_game)
        return pacman_game, pacman_renderer
    elif game_name == "tetris":
        tetris_game = TetrisGame(infinite=infinite, is_ai_controlled=is_ai_controlled)
        tetris_renderer = TetrisPygameRenderer(tetris_game)
        return tetris_game, tetris_renderer
    elif game_name == "racecar":
        racecar_game = RacecarGame(infinite=infinite, is_ai_controlled=is_ai_controlled)
        racecar_renderer = RacecarPygameRenderer(racecar_game)
        return racecar_game, racecar_renderer
    else:
        raise ValueError(f"Unknown game: {game_name}")
