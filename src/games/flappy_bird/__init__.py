from ..registry import register_game
from .game import FlappyGame
from .pygame_renderer import FlappyPygameRenderer

register_game(
    name="flappy",
    game_cls=FlappyGame,
    renderer_cls=FlappyPygameRenderer,
)
