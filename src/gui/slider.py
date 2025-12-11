import pygame
from typing import List
from ..utils.colors import Color


class Slider:
    def __init__(self, frame_rate: int, width: int, height: int):
        self.x: float = (0.05 + (frame_rate / 240)) * width * 0.9
        self.width = width
        self.height = height
        self.area_height = 50
        self.surface = pygame.Surface((width, self.area_height))

    def draw(self) -> None:
        self.surface.fill(Color.BLACK)

        pygame.draw.rect(
            self.surface,
            Color.WHITE,
            pygame.rect.Rect(
                0.05 * self.width,
                0.48 * self.area_height,
                0.90 * self.width,
                0.04 * self.area_height,
            ),
        )

        pygame.draw.circle(
            self.surface,
            Color.WHITE,
            center=(self.x, self.area_height / 2),
            radius=self.area_height / 4,
        )

    def handle_events(self, events: List[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if self.height < my < (self.height + self.area_height) and (
                        self.width * 0.05
                    ) < mx < (self.width * 0.95):
                        self.x = mx

    def get_surface(self) -> pygame.Surface:
        return self.surface

    def compute_frame_rate(self) -> int:
        rel_pos = self.x / self.width
        cur_frame_rate = int((rel_pos - 0.05) * (1 / 0.9) * 235) + 5
        return cur_frame_rate
