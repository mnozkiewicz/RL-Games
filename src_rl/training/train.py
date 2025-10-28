from torch import optim
from ..agents.actor_critic import ActorCriticController
import numpy as np
import matplotlib.pyplot as plt
from ..games.snake.game import SnakeGame
from tqdm import tqdm
from pathlib import Path


def current_path() -> Path:
    current_file_dir = Path(__file__).parent
    return current_file_dir


def main() -> None:
    controller = ActorCriticController(
        state_space_shape=61,
        action_space_size=4,
        batch_size=64,
        hidden_layer_sizes=(1024, 512, 256),
        discount_factor=0.99,
        device="cpu",
        optimizer=optim.AdamW,
        optimizer_kwargs={"lr": 0.0001},
    )

    snake_game = SnakeGame(15, infinite=False)

    num_episodes = 1000
    max_steps_per_episode = 3000

    past_rewards: list[float] = []
    past_lengths: list[int] = []

    current_dir = current_path()

    for i_episode in tqdm(range(num_episodes)):
        snake_game.reset()
        state = snake_game.processed_state()

        reward_sum = 0.0
        steps = 0

        while snake_game.is_running() and steps < max_steps_per_episode:
            action = controller.choose_action(state)
            reward = snake_game.step(action)
            new_state = snake_game.processed_state()
            controller.learn(
                state, action, reward, new_state, not snake_game.is_running()
            )

            state = new_state
            reward_sum += reward
            steps += 1

        snake_length = len(snake_game.get_state().tail)

        past_lengths.append(snake_length)
        past_rewards.append(reward_sum)

        window_size = 10
        # if i_episode % 100 == 0:
        #     torch.save(controller.state_dict(), f"snake_agent_epoch_{i_episode}.pt")

        if i_episode % 50 == 0:
            if len(past_rewards) >= window_size:
                rewards_ma = [
                    np.mean(past_rewards[i : i + window_size])
                    for i in range(len(past_rewards) - window_size)
                ]
                lengths_ma = [
                    np.mean(past_lengths[i : i + window_size])
                    for i in range(len(past_lengths) - window_size)
                ]
                x_vals = list(range(window_size, len(past_rewards)))

                fig, ax1 = plt.subplots()

                color = "tab:green"
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Reward", color=color)
                ax1.plot(x_vals, rewards_ma, color=color)
                ax1.tick_params(axis="y", labelcolor=color)

                ax2 = ax1.twinx()  # create a second y-axis
                color = "tab:blue"
                ax2.set_ylabel("Snake Length", color=color)
                ax2.plot(x_vals, lengths_ma, color=color)
                ax2.tick_params(axis="y", labelcolor=color)

                plt.title("Moving Average of Reward and Snake Length")
                fig.tight_layout()
                plt.savefig(current_dir / f"plots/learning_{i_episode}.png")
                plt.close()

    controller.save_model(str(current_dir / "controller.pt"))


if __name__ == "__main__":
    main()
