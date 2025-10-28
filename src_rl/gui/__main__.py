from .gui import GameGui
from .gui_config import GUIConfig
from ..games.snake.game import SnakeGame
from ..games.snake.renderer import SnakeRenderer

# from ..players.ai_player import AIPlayer
# from ..agents.actor_critic import ActorCriticController

from ..players import HumanPlayer
from ..players.key_maps import snake_key_map


def main() -> None:
    snake_game = SnakeGame(15, infinite=True)
    renderer = SnakeRenderer(snake_game)
    config = GUIConfig(pixel_height=600, pixel_width=600, frame_rate=10)

    # agent = ActorCriticController.load_model("src_rl/experiments/controller.pt")
    # player = AIPlayer(game=snake_game, agent=agent)
    player = HumanPlayer(game=snake_game, key_map=snake_key_map)

    gui = GameGui(renderer=renderer, game=snake_game, config=config, player=player)

    gui.run_game()


if __name__ == "__main__":
    main()
