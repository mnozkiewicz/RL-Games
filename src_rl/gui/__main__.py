import argparse
from pathlib import Path
from typing import Tuple, Dict

# Abstract interfaces
from ..players.base_player import BasePlayer
from ..games.base_game import BaseGame
from ..games.base_renderer import BaseRenderer

# Pygame GUI
from .gui import GameGui
from .gui_config import GUIConfig

# Games and renderers
from ..games.snake.game import SnakeGame
from ..games.snake.renderer import SnakeRenderer
from ..games.flappy_bird.game import FlappyGame
from ..games.flappy_bird.renderer import FlappyRenderer

# Players, agents and key_maps
from ..players import HumanPlayer, AIPlayer
from ..players.key_maps import snake_key_map, flappy_key_map
from ..agents.actor_critic import ActorCriticController


def create_game_and_renderer(
    game_name: str, infinite: bool
) -> Tuple[BaseGame, BaseRenderer[BaseGame], Dict[int, int]]:
    """
    Factory function to initialize the correct game and renderer.
    """
    if game_name == "snake":
        snake_game = SnakeGame(board_size=15, infinite=infinite)
        snake_renderer = SnakeRenderer(snake_game)
        return snake_game, snake_renderer, snake_key_map
    elif game_name == "flappy":
        flappy_game = FlappyGame(infinite=infinite)
        flappy_renderer = FlappyRenderer(flappy_game)
        return flappy_game, flappy_renderer, flappy_key_map
    else:
        raise ValueError(f"Unknown game: {game_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a GUI game with either a human or AI player."
    )

    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        choices=["flappy", "snake"],
        help="Which game to run.",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="ai",
        choices=["human", "ai"],
        help="Control type: human or AI.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Flag telling wheter to use trained agent",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src_rl/training/controller.pt",
        help="Path to the trained AI model.",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
        help="Allow the AI agent to keep learning during play.",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Loop the game evry time the agent or player crashes",
    )

    parser.add_argument(
        "--frame-rate", type=int, default=10, help="Frames per second for the GUI."
    )
    parser.add_argument(
        "--pixel-size", type=int, default=600, help="Window size in pixels."
    )

    args = parser.parse_args()

    # Game and GUI setup
    game, renderer, key_map = create_game_and_renderer(args.game, args.infinite)
    config = GUIConfig(
        pixel_height=args.pixel_size,
        pixel_width=args.pixel_size,
        frame_rate=args.frame_rate,
    )

    # Player setup
    print(f"Starting game {args.game} as {args.player}")

    player: BasePlayer
    if args.player == "human":
        player = HumanPlayer(game=game, key_map=key_map)
    else:
        if args.pretrained:
            if not Path(args.model_path).exists():
                raise ValueError(f"There is no model in path {args.model_path}")
            print(f"Loading AI model from {args.model_path}...")
            agent = ActorCriticController.load_model(args.model_path)
        else:
            agent = ActorCriticController(
                game.processed_state().shape[0],
                game.number_of_moves,
                hidden_layer_sizes=(512, 256, 64),
            )

        player = AIPlayer(game=game, agent=agent, learn=args.learn)

    # Run the GUI
    gui = GameGui(renderer=renderer, game=game, config=config, player=player)
    gui.run_game()


if __name__ == "__main__":
    main()
