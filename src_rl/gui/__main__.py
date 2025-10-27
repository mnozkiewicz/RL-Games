from .gui import GameGui
from .gui_config import GUIConfig
from ..games.snake.game import SnakeGame
from ..games.snake.renderer import SnakeRenderer

# from .players.ai_player import AIPlayer
# from .agents.dummy_agent import DummyAgent
from ..players.human_player import HumanPlayer


def main() -> None:
    snake_game = SnakeGame(15, infinite=True)
    renderer = SnakeRenderer(snake_game)
    config = GUIConfig(pixel_height=600, pixel_width=600, frame_rate=10)
    # player = AIPlayer(game=snake_game, agent=DummyAgent(61, 4))
    player = HumanPlayer(game=snake_game)

    gui = GameGui(renderer=renderer, game=snake_game, config=config, player=player)

    gui.run_game()


if __name__ == "__main__":
    main()
