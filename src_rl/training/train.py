from torch import optim
from ..agents.actor_critic import ActorCriticController
from ..agents.base_agent import BaseAgent
from ..games.base_game import BaseGame
from ..games.snake.game import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import argparse
from tqdm import tqdm


def get_current_dir_path() -> Path:
    current_file_dir = Path(__file__).parent
    return current_file_dir


def plot_metrics(
    past_rewards: List[float],
    past_scores: List[int],
    window_size: int,
    current_dir: Path,
    episode_number: int,
) -> None:
    rewards_ma = [
        np.mean(past_rewards[i : i + window_size])
        for i in range(len(past_rewards) - window_size)
    ]
    lengths_ma = [
        np.mean(past_scores[i : i + window_size])
        for i in range(len(past_scores) - window_size)
    ]

    x_vals = list(range(window_size, len(past_rewards)))
    fig, ax1 = plt.subplots()

    color = "tab:green"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color=color)
    ax1.plot(x_vals, rewards_ma, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Snake Length", color=color)
    ax2.plot(x_vals, lengths_ma, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Moving Average of Reward and Snake Length")
    fig.tight_layout()
    plt.savefig(current_dir / f"plots/learning_{episode_number}.png")
    plt.close()


def train(
    agent: BaseAgent,
    game: BaseGame,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 3000,
    graph_plotting_step: int = 25,
    window_size: int = 10,  # To smooth out the reward plot
) -> BaseAgent:
    past_rewards: List[float] = []
    past_scores: List[int] = []

    current_dir = get_current_dir_path()

    for i_episode in tqdm(range(num_episodes)):
        game.reset()
        state = game.processed_state()

        reward_sum = 0.0
        maximum_score = 0
        steps = 0

        while game.is_running() and steps < max_steps_per_episode:
            action = agent.choose_action(state)

            reward = game.step(action)
            new_state = game.processed_state()
            terminal = not game.is_running()
            agent.learn(state, action, reward, new_state, terminal)

            state = new_state

            maximum_score = max(maximum_score, game.score())
            reward_sum += reward
            steps += 1

        past_scores.append(maximum_score)
        past_rewards.append(reward_sum)

        # if i_episode % 100 == 0: Should probably add saving step
        #     torch.save(controller.state_dict(), f"snake_agent_epoch_{i_episode}.pt")

        if (i_episode + 1) % graph_plotting_step == 0:
            if len(past_rewards) >= window_size:
                plot_metrics(
                    past_rewards, past_scores, window_size, current_dir, i_episode + 1
                )

    agent.save_model(str(current_dir / "controller.pt"))

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Actor-Critic agent")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes."
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="Maximum steps per episode."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run training on.",
    )
    parser.add_argument(
        "--plot-step", type=int, default=25, help="How often to plot metrics."
    )
    parser.add_argument(
        "--window", type=int, default=10, help="Moving average window size."
    )
    parser.add_argument("--board-size", type=int, default=15, help="Snake board size.")

    args = parser.parse_args()

    snake_game = SnakeGame(args.board_size, infinite=False)

    actor_critic_agent = ActorCriticController(
        state_space_shape=snake_game.processed_state().shape[0],
        action_space_size=snake_game.number_of_moves,
        batch_size=args.batch_size,
        hidden_layer_sizes=(1024, 512, 256),
        discount_factor=0.99,
        device=args.device,
        optimizer=optim.AdamW,
        optimizer_kwargs={"lr": args.lr},
    )

    train(
        actor_critic_agent,
        snake_game,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        graph_plotting_step=args.plot_step,
        window_size=args.window,
    )


if __name__ == "__main__":
    main()
