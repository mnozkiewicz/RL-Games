from ..registry import register_game
from .game import TetrisGame
from .pygame_renderer import TetrisPygameRenderer

register_game(
    name="tetris",
    game_cls=TetrisGame,
    renderer_cls=TetrisPygameRenderer,
)
