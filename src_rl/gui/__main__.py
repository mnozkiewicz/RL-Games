from .gui import GameGui
from .gui_config import GUIConfig
from ..games.snake.game import SnakeGame
from ..games.snake.renderer import SnakeRenderer
# from ..games.flappy_bird.game import FlappyGame
# from ..games.flappy_bird.renderer import FlappyRenderer

from ..players.ai_player import AIPlayer
from ..agents.actor_critic import ActorCriticController

# from ..players import HumanPlayer
# from ..players.key_maps import snake_key_map


def main() -> None:
    game = SnakeGame(15, infinite=True)
    renderer = SnakeRenderer(game)
    config = GUIConfig(pixel_height=600, pixel_width=600, frame_rate=10)

    agent = ActorCriticController.load_model("src_rl/training/controller.pt")
    player = AIPlayer(game=game, agent=agent)
    # player = HumanPlayer(game=game, key_map=snake_key_map)

    gui = GameGui(renderer=renderer, game=game, config=config, player=player)

    gui.run_game()


if __name__ == "__main__":
    main()
