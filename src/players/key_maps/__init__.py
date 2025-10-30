"""
Every game should have key map defined,
with action -1 being ignored.
"""

import pygame
from typing import Dict
from collections import defaultdict


def get_game_key_map(game_name: str) -> Dict[int, int]:
    """
    Factory function to initialize the correct game.
    """
    if game_name == "snake":
        return snake_key_map
    elif game_name == "flappy":
        return flappy_key_map
    else:
        raise ValueError(f"Unknown game: {game_name}")


snake_key_map = defaultdict(
    lambda: -1,
    {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_w: 0,
        pygame.K_s: 1,
        pygame.K_a: 2,
        pygame.K_d: 3,
    },
)


flappy_key_map = defaultdict(
    lambda: -1,
    {
        pygame.K_DOWN: 0,
        pygame.K_UP: 1,
        pygame.K_s: 0,
        pygame.K_w: 1,
    },
)


__all__ = ["snake_key_map", "flappy_key_map"]
