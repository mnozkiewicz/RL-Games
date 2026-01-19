from ..registry import register_game
from .game import SnakeGame
from .pygame_renderer import SnakePygameRenderer

register_game(
    name="snake",
    game_cls=SnakeGame,
    renderer_cls=SnakePygameRenderer,
)
