from games import GUIConfig, GameGui
from games.snake.engine import SnakeGame


def main() -> None:
    snake_game = SnakeGame(15, infinite=True)

    config = GUIConfig(pixel_height=600, pixel_width=600, frame_rate=10)

    gui = GameGui(game=snake_game, config=config)

    gui.run_game()


if __name__ == "__main__":
    main()
