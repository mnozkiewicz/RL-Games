import pygame
from ...utils.colors import Color
from .game import SnakeGame
from .utils import Pos
from ..base_pygame_renderer import BasePygameRenderer
from pathlib import Path


class SnakePygameRenderer(BasePygameRenderer[SnakeGame]):
    def __init__(self, game: SnakeGame):
        super().__init__(game)

    def load_image(self, path: Path) -> pygame.Surface:
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (self.cell_width, self.cell_height))

    def rotate(self, image: pygame.Surface, angle: int) -> pygame.Surface:
        return pygame.transform.rotate(image, angle)

    def load_assets(self) -> None:
        # Path to folder containing assets
        asset_path = Path("assets/snake")

        self.apple = self.load_image(asset_path / "apple.png")
        self.head = self.load_image(asset_path / "head.png")
        self.body = self.load_image(asset_path / "body.png")
        self.curve = self.load_image(asset_path / "turn.png")
        self.tail = self.load_image(asset_path / "tail.png")

    def init_pygame_renderer(self, surface: pygame.Surface) -> None:
        screen_width = surface.get_width()
        screen_height = surface.get_height()

        self.cell_width = screen_width // self.game.board_size
        self.cell_height = screen_height // self.game.board_size

        self.load_assets()

    def compute_vector(self, pos1: Pos, pos2: Pos, mod: int) -> Pos:
        diff = (pos1 - pos2).mod_index(mod)
        x, y = diff
        x = -1 if x == mod - 1 else x
        y = -1 if y == mod - 1 else y
        return Pos(x, y)

    def compute_turn_angle(self, vec1: Pos, vec2: Pos) -> int:
        if {vec1, vec2} == {Pos(-1, 0), Pos(0, 1)}:
            return 90
        if {vec1, vec2} == {Pos(0, 1), Pos(1, 0)}:
            return 180
        if {vec1, vec2} == {Pos(1, 0), Pos(0, -1)}:
            return -90
        return 0

    def compute_tail_angle(self, vec: Pos) -> int:
        if vec == Pos(1, 0):
            return 0
        if vec == Pos(0, -1):
            return 90
        if vec == Pos(-1, 0):
            return 180
        return -90

    def draw(self, surface: pygame.Surface) -> None:
        state = self.game.get_state()

        if not state.running:
            surface.fill(Color.BLACK)
            return

        surface.fill(Color.LIGHTGREEN)

        food = state.food
        surface.blit(self.apple, (food.x * self.cell_width, food.y * self.cell_height))

        state.snake_dir
        head = state.head
        surface.blit(
            self.rotate(self.head, state.snake_dir.angle()),
            (head.x * self.cell_width, head.y * self.cell_height),
        )

        if len(state.tail) < 2:
            return

        size = state.board_size

        for next_pos, pos, prev_pos in zip(
            state.tail[:-2], state.tail[1:-1], state.tail[2:]
        ):
            # Center prev and next positions around the currently considered part of snake's body
            vec1 = self.compute_vector(next_pos, pos, size)
            vec2 = self.compute_vector(prev_pos, pos, size)

            # If dot-product equals 0 then it must be 90 degree angle (turn)
            if (vec1.x * vec2.x + vec1.y * vec2.y) == 0:
                turn_angle = self.compute_turn_angle(vec1, vec2)
                surface.blit(
                    self.rotate(self.curve, turn_angle),
                    (pos.x * self.cell_width, pos.y * self.cell_height),
                )
            else:
                angle = 90 if (prev_pos - next_pos).y != 0 else 0
                surface.blit(
                    self.rotate(self.body, angle),
                    (pos.x * self.cell_width, pos.y * self.cell_height),
                )

        tail = state.tail[-1]
        diff = self.compute_vector(state.tail[-2], tail, size)
        tail_angle = self.compute_tail_angle(diff)
        surface.blit(
            self.rotate(self.tail, tail_angle),
            (tail.x * self.cell_width, tail.y * self.cell_height),
        )
