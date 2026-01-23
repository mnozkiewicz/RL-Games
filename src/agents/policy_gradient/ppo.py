from __future__ import annotations
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Union, Literal, Dict, Any
from .trajectory_buffer import TrajectoryBuffer
from ..base_agent import BaseAgent


class PPOAgent(BaseAgent, nn.Module):
    def __init__(
        self,
        state_space_shape: Union[Tuple[int, ...], int],
        action_space_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        entropy_coef: float = 0.01,
        device: str = "cpu",
        input_type: Literal["raw_pixels", "processed_state"] = "processed_state",
    ) -> None:
        super().__init__()

        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.input_type = input_type
        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes

        # PPO Hyperparameters
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = batch_size
        self.entropy_coef = entropy_coef

        # Additional training attributes
        self.last_action_log_prob = 0.0

        # Network Setup
        if input_type == "processed_state":
            if isinstance(state_space_shape, tuple):
                assert len(state_space_shape) == 1
            else:
                state_space_shape = (state_space_shape,)
            self.model = self._create_networks(
                *state_space_shape, hidden_layer_sizes, action_space_size
            ).to(device)
        else:
            assert isinstance(state_space_shape, tuple) and len(state_space_shape) == 3
            self.model = self._create_cnn_networks(
                state_space_shape, action_space_size
            ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.trajectory_buffer = TrajectoryBuffer(max_len=buffer_size)

    def _create_networks(
        self, input_size: int, hidden_layer_sizes: Tuple[int, ...], output_size: int
    ) -> nn.ModuleDict:
        """
        Create actor and critic networks.
        Actor outputs a probability distribution over actions.
        Critic outputs a single value estimate of the state.
        """

        # Actor
        actor_layers: List[nn.Module] = []
        in_features = input_size

        for hidden_size in hidden_layer_sizes:
            actor_layers.append(nn.Linear(in_features, hidden_size))
            actor_layers.append(nn.LayerNorm(hidden_size))  # Normalize activations
            actor_layers.append(nn.ReLU())
            in_features = hidden_size

        actor_layers.append(nn.Linear(in_features, output_size))
        actor_layers.append(nn.Softmax(dim=-1))  # Probability distribution over actions

        actor = nn.Sequential(*actor_layers)

        # Critic Network
        critic_layers: List[nn.Module] = []
        in_features = input_size

        for hidden_size in hidden_layer_sizes:
            critic_layers.append(nn.Linear(in_features, hidden_size))
            critic_layers.append(nn.LayerNorm(hidden_size))  # LayerNorm for stability
            critic_layers.append(nn.ReLU())
            in_features = hidden_size

        critic_layers.append(nn.Linear(in_features, 1))  # Single value estimate
        critic = nn.Sequential(*critic_layers)

        return nn.ModuleDict({"actor": actor, "critic": critic})

    def _create_cnn_networks(
        self, input_shape: Tuple[int, int, int], output_size: int
    ) -> nn.ModuleDict:
        input_channels, input_height, input_width = input_shape

        def cnn_out_dim(cnn: nn.Module) -> int:
            with torch.no_grad():
                dummy = torch.zeros(1, input_channels, input_height, input_width)
                return int(cnn(dummy).shape[1])

        backbone_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LayerNorm([32, (input_height - 8) // 4 + 1, (input_width - 8) // 4 + 1]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LayerNorm(
                [
                    64,
                    ((input_height - 8) // 4 - 4) // 2 + 1,
                    ((input_width - 8) // 4 - 4) // 2 + 1,
                ]
            ),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        actor_fc = nn.Sequential(
            nn.Linear(cnn_out_dim(backbone_cnn), 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Softmax(dim=-1),
        )
        actor = nn.Sequential(backbone_cnn, actor_fc)

        critic_fc = nn.Sequential(
            nn.Linear(cnn_out_dim(backbone_cnn), 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        critic = nn.Sequential(backbone_cnn, critic_fc)

        return nn.ModuleDict(
            {
                "actor": actor,
                "critic": critic,
            }
        )

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action given a state.
        """
        state_tensor = torch.Tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.model["actor"](state_tensor)

        if self.eval_mode:
            action = torch.argmax(probs, dim=1)
            return int(action.item())

        distribution = Categorical(probs)
        action = distribution.sample()  # type: ignore
        log_prob = distribution.log_prob(action)  # type: ignore

        if not isinstance(log_prob, torch.Tensor):
            raise RuntimeError()

        self.last_action_log_prob = float(log_prob.item())
        return int(action.item())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        self.trajectory_buffer.store(
            state, action, reward, next_state, terminal, self.last_action_log_prob
        )
        if self.trajectory_buffer.is_full():
            self._update()
            self.trajectory_buffer.clear()

    @torch.no_grad()
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes Generalized Advantage Estimation (GAE) entirely in PyTorch.
        """
        deltas = rewards + self.discount_factor * next_values * masks - values
        advantages = torch.zeros_like(deltas)
        gae = torch.tensor(0.0)

        for i in reversed(range(len(rewards))):
            gae = deltas[i] + (self.discount_factor * self.gae_lambda * masks[i] * gae)
            advantages[i] = gae

        return advantages

    def _update(self) -> None:
        states, actions, rewards, next_states, dones, old_log_probs = (
            self.trajectory_buffer.get_batch()
        )

        states_tensor = torch.tensor(states).to(self.device)
        next_states_tensor = torch.tensor(next_states).to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )
        masks_tensor = torch.tensor(1 - dones, dtype=torch.float32, device=self.device)

        # Compute Values (Current Estimate) for GAE
        with torch.no_grad():
            values: torch.Tensor = self.model["critic"](states_tensor).squeeze()
            next_values: torch.Tensor = self.model["critic"](
                next_states_tensor
            ).squeeze()

        # Calculate GAE
        advantages = self._compute_gae(
            rewards_tensor, values, next_values, masks_tensor
        )

        # Compute returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update Loop
        dataset_size = len(states)
        indices = torch.randperm(dataset_size)

        for _ in range(self.ppo_epochs):
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]

                # Mini-batch slices
                mb_states = states_tensor[idx]
                mb_actions = actions_tensor[idx]
                mb_old_log_probs = old_log_probs_tensor[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Forward pass
                probs = self.model["actor"](mb_states)
                curr_values = self.model["critic"](mb_states).squeeze()

                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)  # type: ignore

                if not isinstance(new_log_probs, torch.Tensor):
                    raise RuntimeError()

                entropy = dist.entropy().mean()  # type: ignore

                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Surrogate Loss
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * mb_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (curr_values - mb_returns).pow(2).mean()  # MSE

                total_loss = actor_loss + critic_loss - (self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def set_eval_mode(self) -> None:
        self.eval()
        self.eval_mode = True

    def set_train_mode(self) -> None:
        self.train()
        self.eval_mode = False

    def save_model(self, path: str) -> None:
        """
        Save model configuration and weights.
        Stores:
        - config.json : network architecture and state/action dimensions
        - model.pth   : PyTorch model weights
        """
        path = f"{path}_ppo_{self.input_type}"
        print(f"Saving model to {path}...")
        os.makedirs(path, exist_ok=True)

        config: Dict[str, Any] = {
            "state_space_shape": self.state_space_shape,
            "action_space_size": self.action_space_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "input_type": self.input_type,
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.model.to("cpu").state_dict(), os.path.join(path, "model.pth"))

    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> PPOAgent:
        """
        Load a model from disk.
        Recreates the architecture using config.json and loads the trained weights.
        """

        print(f"Loading model from {path}...")

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        config["state_space_shape"] = tuple(config["state_space_shape"])
        config["hidden_layer_sizes"] = tuple(config["hidden_layer_sizes"])
        agent = cls(device=device, **config)

        weights_path = os.path.join(path, "model.pth")
        agent.model.load_state_dict(
            torch.load(weights_path, map_location=torch.device(device))
        )

        print("Model loaded successfully.")
        return agent
