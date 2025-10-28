import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Optional, Any, Type, Tuple, List, Dict
from .trajectory_buffer import TrajectoryBuffer
from ..base_agent import BaseAgent


class ActorCriticController(BaseAgent, nn.Module):
    def __init__(
        self,
        state_space_shape: int,
        action_space_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        batch_size: int = 64,
        discount_factor: float = 0.99,
        device: str = "cpu",
        optimizer: Optional[Type[optim.Optimizer]] = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(state_space_shape, action_space_size)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.discount_factor = discount_factor
        self.device = device
        self.trajectory_buffer = TrajectoryBuffer(batch_size)

        self.model = self._create_networks(
            state_space_shape, hidden_layer_sizes, action_space_size
        ).to(device)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

    def _create_networks(
        self, input_size: int, hidden_layer_sizes: Tuple[int, ...], output_size: int
    ) -> nn.ModuleDict:
        # --- Build Actor Network ---
        actor_layers: List[nn.Module] = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_layer_sizes:
            actor_layers.append(nn.Linear(in_features, hidden_size))
            actor_layers.append(nn.LayerNorm(hidden_size))
            actor_layers.append(nn.ReLU())
            in_features = hidden_size  # Update in_features for the next layer

        # Add output layer
        actor_layers.append(nn.Linear(in_features, output_size))
        actor_layers.append(nn.Softmax(dim=-1))  # Outputs a probability distribution

        actor = nn.Sequential(*actor_layers)

        # --- Build Critic Network ---
        critic_layers: List[nn.Module] = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_layer_sizes:
            critic_layers.append(nn.Linear(in_features, hidden_size))
            actor_layers.append(nn.LayerNorm(hidden_size))
            critic_layers.append(nn.ReLU())
            in_features = hidden_size  # Update in_features for the next layer

        # Add output layer
        critic_layers.append(nn.Linear(in_features, 1))  # Outputs a single value
        critic = nn.Sequential(*critic_layers)

        return nn.ModuleDict({"actor": actor, "critic": critic})

    def choose_action(self, state: np.ndarray) -> int:
        state_tensor = (
            torch.tensor(state).reshape(1, self.state_space_shape).to(self.device)
        )

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

    def save_model(self, path: str):
        """Saves the model's weights and its configuration."""
        print(f"Saving model to {path}...")
        # 1. Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # 2. Save model configuration
        config: Dict[str, Any] = {
            "state_space_shape": self.state_space_shape,
            "action_space_size": self.action_space_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

        # 3. Save model weights
        torch.save(self.model.to("cpu").state_dict(), os.path.join(path, "model.pth"))

    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        """Loads a model from a saved configuration and weights."""
        print(f"Loading model from {path}...")

        # 1. Load model configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        # 2. Re-create the agent with the loaded config
        # We must convert shape from list (JSON) to tuple
        config["hidden_layer_sizes"] = tuple(config["hidden_layer_sizes"])
        agent = cls(device=device, **config)

        # 3. Load model weights
        weights_path = os.path.join(path, "model.pth")
        agent.model.load_state_dict(
            torch.load(weights_path, map_location=torch.device(device))
        )

        print("Model loaded successfully.")
        return agent
