import argparse
from pathlib import Path

# Abstract interfaces
from ..players.base_player import BasePlayer

# Pygame GUI
from .gui import GameGui
from .gui_config import GUIConfig

# Players and agents
from ..players import HumanPlayer, AIPlayer
from ..agents.policy_gradient import ActorCriticAgent

# Game object factory
from ..games.registry import create_game_and_renderer, GAME_REGISTRY

GAMES = list(GAME_REGISTRY.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a GUI game with either a human or AI player."
    )

    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        choices=GAMES,
        help="Which game to run.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="processed_state",
        choices=["processed_state", "raw_pixels"],
        help="Should agent make choices based on a game image or a processed_state.",
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
        help="Flag telling wheter to use the trained agent",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
        help="Allow the AI agent to keep learning during play.",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Start over the game every time the agent or player crashes",
    )

    parser.add_argument(
        "--frame-rate", type=int, default=10, help="Frames per second for the GUI."
    )
    parser.add_argument(
        "--pixel-size", type=int, default=600, help="Window size in pixels."
    )

    args = parser.parse_args()

    # Game and GUI setup
    game, renderer = create_game_and_renderer(
        args.game, args.input_type, args.infinite, (args.player == "ai")
    )
    key_map = renderer.get_key_map()
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
            model_path = f"weights/{game.name()}_controller_{args.input_type}"
            if not Path(model_path).exists():
                raise ValueError(f"There is no model in path {model_path}")
            print(f"Loading AI model from {model_path}...")
            agent = ActorCriticAgent.load_model(model_path)
        else:
            agent = ActorCriticAgent(
                game.state().shape,
                game.number_of_moves,
                hidden_layer_sizes=(64, 64),
                input_type=args.input_type,
            )

        player = AIPlayer(game=game, agent=agent, learn=args.learn)

    # Run the GUI
    gui = GameGui(renderer=renderer, game=game, config=config, player=player)
    gui.run_game()


if __name__ == "__main__":
    main()
