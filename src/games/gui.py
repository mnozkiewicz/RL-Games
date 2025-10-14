import pygame
# from game_engine import SnakeGame, State, Action
# from .players import BasePlayer


class Color:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class GameGui:
    def __init__(
        self,
        # player: BasePlayer,
        pixel_size: int = 600,
        board_size: int = 20,
        frame_rate: int = 10,
    ):
        self.pixel_size: int = pixel_size
        self.board_size: int = board_size
        self.cell_size: int = pixel_size // board_size
        self.frame_rate: int = frame_rate

        assert 5 <= frame_rate <= 120, "Frame rate should be in range [5, 120]"
        assert pixel_size % board_size == 0, (
            "Height and width in pixel should be divisible by height in cells"
        )

        self.screen: pygame.Surface
        self.clock: pygame.time.Clock

        # self.snake_game: SnakeGame = SnakeGame(self.board_size)
        # self.player: BasePlayer = player

        # self.state: State
        self.running: bool = True

    def create_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(
            x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
        )

    # def draw_board(self):
    #     self.state = self.snake_game.state()

    #     if not self.state.running:
    #         self.endgame()
    #         return

    #     self.screen.fill(Color.WHITE)
    #     for pos in self.state.tail:
    #         rect = self.create_rect(pos.x, pos.y)
    #         pygame.draw.rect(self.screen, Color.GREEN, rect)

    #     head = self.state.head
    #     rect = self.create_rect(head.x, head.y)
    #     pygame.draw.rect(self.screen, Color.RED, rect)

    #     food = self.state.food
    #     rect = self.create_rect(food.x, food.y)
    #     pygame.draw.rect(self.screen, Color.BLUE, rect)

    # def endgame(self):
    #     self.screen.fill(Color.BLACK)

    def check_if_quit(self, events: list[pygame.event.Event]):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

    def handle_events(self) -> None:
        events = list(pygame.event.get())
        self.check_if_quit(events)
        # return self.player.move(events, self.snake_game.state())

    def run_game(self) -> None:
        pygame.init()

        self.running = True
        self.screen = pygame.display.set_mode((self.pixel_size, self.pixel_size))
        self.screen.fill(Color.WHITE)

        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        while self.running:
            # events = self.handle_events()
            # self.snake_game.game_step(events)
            # self.draw_board()
            events = list(pygame.event.get())
            self.check_if_quit(events)

            pygame.display.flip()
            self.clock.tick(self.frame_rate)

            # running = running and self.snake_game.state().running

        pygame.quit()


def main() -> None:
    snake_gui = GameGui(pixel_size=600, board_size=20, frame_rate=120)

    snake_gui.run_game()


if __name__ == "__main__":
    main()
