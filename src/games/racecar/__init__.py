from ..registry import register_game
from .game import RacecarGame
from .pygame_renderer import RacecarPygameRenderer

register_game(
    name="racecar",
    game_cls=RacecarGame,
    renderer_cls=RacecarPygameRenderer,
)
