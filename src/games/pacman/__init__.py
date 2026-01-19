from ..registry import register_game
from .game import PacmanGame
from .pygame_renderer import PacmanPygameRenderer

register_game(name="pacman", game_cls=PacmanGame, renderer_cls=PacmanPygameRenderer)
