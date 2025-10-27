import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .trajectory_buffer import TrajectoryBuffer
from ..base_agent import BaseAgent


class ActorCriticController(BaseAgent, nn.Module):
    def __init__(
        self,
        state_space_shape: int,
        action_space_size: int,
        batch_size: int,
        actor: nn.Module,
        critic: nn.Module,
        learning_rate: float = 0.00001,
        discount_factor: float = 0.99,
        device: str = "cpu",
    ) -> None:
        super().__init__(state_space_shape, action_space_size)

        self.discount_factor = discount_factor
        self.device = device
        self.trajectory_buffer = TrajectoryBuffer(batch_size)

        self.model = nn.ModuleDict({"actor": actor, "critic": critic}).to(device)

        self._check_network_dimensions(state_space_shape, action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _check_network_dimensions(self, input_size: int, output_size: int) -> None:
        dummy_input = torch.zeros(
            10, input_size, device=self.device, dtype=torch.float32
        )

        try:
            with torch.no_grad():
                actor_output = self.model["actor"](dummy_input)
                critic_output = self.model["critic"](dummy_input)
        except Exception as e:
            raise RuntimeError(
                f"Error when running dummy input through actor/critic networks: {e}"
                f"Possibly missmatch involving the `input_state_dim` or `number of actions`"
            )

        # Actor should output (batch_size, number_of_actions)
        if actor_output.shape != (10, output_size):
            raise ValueError(
                f"Actor network output shape {actor_output.shape} does not match "
                f"expected (1, {output_size}). "
                f"Check the final layer of your actor network."
            )

        # Critic should output (batch_size, 1)
        if critic_output.shape != (10, 1):
            raise ValueError(
                f"Critic network output shape {critic_output.shape} does not match (1, 1). "
                f"Check the final layer of your critic network."
            )

    def choose_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state.flatten()).reshape(1, -1).to(self.device)

        with torch.no_grad():
            probs = self.model["actor"](state_tensor)

        if self.eval_mode:
            # Deterministic (greedy) action
            action = torch.argmax(probs, dim=1)
            return int(action.item())
        else:
            # Stochastic (sampling) action
            distribution = Categorical(probs=probs)
            action = distribution.sample()
            self.log_prob = distribution.log_prob(action)
            return int(action.item())

    @torch.no_grad()
    def evaluate_state(self, state: torch.Tensor) -> torch.Tensor:
        expected_reward = self.model["critic"](state)
        return expected_reward.detach()

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        self.trajectory_buffer.store(state, action, reward, next_state, terminal)

        if self.trajectory_buffer.is_full():
            states, actions, rewards, next_states, dones = (
                self.trajectory_buffer.get_batch()
            )

            # Numpy to tensor conversions
            states_tensor = torch.tensor(states).to(self.device)
            next_states_tensor = torch.tensor(next_states).to(self.device)
            rewards_tensor = torch.tensor(rewards.reshape(rewards.shape[0], 1)).to(
                self.device
            )

            # Clear all gradients
            self.optimizer.zero_grad()

            # Critic
            critics_evaluation: torch.Tensor = self.model["critic"](states_tensor)
            remaining_reward = self.discount_factor * self.evaluate_state(
                next_states_tensor
            )
            remaining_reward[dones] = 0.0

            # Actor
            probs = self.model["actor"](states_tensor)
            distribution = Categorical(probs=probs)
            actions_tensor: torch.Tensor = torch.tensor(
                actions, dtype=torch.long, device=self.device
            )
            log_probs: torch.Tensor = distribution.log_prob(actions_tensor)  # type: ignore

            if not isinstance(log_probs, torch.Tensor):
                raise RuntimeError("Error while computing the distribution")

            # Cost function
            difference = rewards_tensor + remaining_reward - critics_evaluation
            critics_loss = difference.square()
            actors_loss = -difference.detach() * log_probs.reshape(-1, 1)

            total_loss = (critics_loss + actors_loss).mean()
            total_loss.backward()
            self.optimizer.step()

            self.trajectory_buffer.clear()

    def set_eval_mode(self):
        self.eval()
        self.eval_mode = True

    def set_train_mode(self):
        self.train()
        self.eval_mode = False
