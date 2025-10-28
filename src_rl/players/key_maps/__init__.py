"""
Every game should have key map defined,
with action -1 being ignored.
"""

import pygame
from collections import defaultdict

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
