from typing import Dict, Type, Tuple, Literal
from ..games.base_game import BaseGame
from .base_pygame_renderer import BasePygameRenderer

GAME_REGISTRY: Dict[str, Type[BaseGame]] = {}
RENDERER_REGISTRY: Dict[str, Type[BasePygameRenderer[BaseGame]]] = {}


def register_game(
    name: str,
    game_cls: Type[BaseGame],
    renderer_cls: Type[BasePygameRenderer[BaseGame]],
) -> None:
    if name in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' already registered")

    GAME_REGISTRY[name] = game_cls
    RENDERER_REGISTRY[name] = renderer_cls


def create_game_and_renderer(
    game_name: str,
    state_type: Literal["processed_state", "raw_pixels"],
    infinite: bool,
    is_ai_controlled: bool,
) -> Tuple[BaseGame, BasePygameRenderer[BaseGame]]:
    """
    Factory function to initialize the correct game and renderer.
    """
    if game_name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {game_name}")

    game = GAME_REGISTRY[game_name](
        state_type=state_type, infinite=infinite, is_ai_controlled=is_ai_controlled
    )
    renderer = RENDERER_REGISTRY[game_name](game)
    return game, renderer


def create_game_engine(
    game_name: str, state_type: Literal["processed_state", "raw_pixels"], infinite: bool
) -> BaseGame:
    """
    Factory function to initialize the correct game engine.
    """
    if game_name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {game_name}")

    game = GAME_REGISTRY[game_name](
        state_type=state_type,
        infinite=infinite,
    )
    return game
