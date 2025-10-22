from .utils.gui import GameGui
from .utils.gui_config import GUIConfig
from .games.snake.game import SnakeGame
from .games.snake.renderer import SnakeRenderer


def main() -> None:
    snake_game = SnakeGame(15, infinite=True)
    renderer = SnakeRenderer(snake_game)
    config = GUIConfig(pixel_height=600, pixel_width=600, frame_rate=10)
    gui = GameGui(renderer=renderer, game=snake_game, config=config)

    gui.run_game()


if __name__ == "__main__":
    main()
