from __future__ import annotations
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Optional, Any, Type, Tuple, List, Dict, Union, Literal
from .trajectory_buffer import TrajectoryBuffer
from ..base_agent import BaseAgent


class ActorCriticAgent(BaseAgent, nn.Module):
    """
    Actor-Critic agent for reinforcement learning.
    """

    def __init__(
        self,
        state_space_shape: Union[Tuple[int, ...], int],
        action_space_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        batch_size: int = 64,
        epsilon: float = 0.00,  # Exploration factor for epsilon-greedy
        discount_factor: float = 0.99,
        device: str = "cpu",
        optimizer: Optional[Type[optim.Optimizer]] = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        input_type: Literal["raw_pixels", "processed_state"] = "processed_state",
    ) -> None:
        super().__init__()

        if input_type == "processed_state":
            if isinstance(state_space_shape, tuple):
                assert len(state_space_shape) == 1, (
                    "For Actor Critic Agent with `input_type` == `processed_state`"
                    "state_space_shape can be either an integer or one-element tuple"
                )
            else:
                state_space_shape = (state_space_shape,)
            # Create actor and critic networks
            self.model = self._create_networks(
                *state_space_shape, hidden_layer_sizes, action_space_size
            ).to(device)

        else:
            raise NotImplementedError("Yet to implement the raw pixels version")
            # assert isinstance(state_space_shape, tuple) and len(state_space_shape) == 3, (
            #     "For Actor Critic Agent with `input_type` == `processed_state`"
            #     "state_space_shape can be either an integer or one-element tuple"
            # )
            # self.model = self._create_networks(
            #     state_space_shape[0], hidden_layer_sizes, action_space_size
            # ).to(device)

        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.eval_mode = False

        # Agent hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.discount_factor = discount_factor
        self.device = device
        self.epsilon = epsilon

        # Buffer to store trajectories before batch updates
        assert batch_size >= 2, "Size of batch should be at least 2"
        self.trajectory_buffer = TrajectoryBuffer(batch_size)

        # Set optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.optimizer: optim.Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

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

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action given a state.
        """
        state_tensor = (
            torch.tensor(state).reshape(1, *self.state_space_shape).to(self.device)
        )

        with torch.no_grad():
            probs = self.model["actor"](state_tensor)

        action: torch.Tensor
        if self.eval_mode:
            # Greedy action for evaluation
            action = torch.argmax(probs, dim=1)
            return int(action.item())
        else:
            # Apply epsilon-greedy exploration if epsilon > 0
            if self.epsilon > 0:
                probs = probs * (1 - self.epsilon) + self.epsilon / float(
                    probs.view(-1).shape[0]
                )

            # Sample action from the categorical distribution
            distribution = Categorical(probs=probs)
            action = distribution.sample()  # type: ignore[no-untyped-call]
            return int(action.item())

    @torch.no_grad()
    def evaluate_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate the value of a state using the critic network.
        """
        expected_reward: torch.Tensor = self.model["critic"](state)
        return expected_reward.detach()

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        """
        Store a transition in the trajectory buffer.
        Once the buffer is full, update actor and critic networks.
        """

        # Store transition
        self.trajectory_buffer.store(state, action, reward, next_state, terminal)

        if self.trajectory_buffer.is_full():
            # Retrieve batch from buffer
            states, actions, rewards, next_states, dones = (
                self.trajectory_buffer.get_batch()
            )

            # Convert to tensors
            states_tensor = torch.tensor(states).to(self.device)
            next_states_tensor = torch.tensor(next_states).to(self.device)
            rewards_tensor = torch.tensor(rewards.reshape(rewards.shape[0], 1)).to(
                self.device
            )
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Critic state value estimation before taking an action
            critics_evaluation: torch.Tensor = self.model["critic"](states_tensor)

            # Critic state value estimation after taking an action and receiving reward
            # Uses bootstraping
            remaining_reward = self.discount_factor * self.evaluate_state(
                next_states_tensor
            )
            remaining_reward[dones] = 0.0  # Zero out terminal states

            # Compute log probabilities
            probs = self.model["actor"](states_tensor)
            distribution = Categorical(probs=probs)
            log_probs: Any = distribution.log_prob(actions_tensor)  # type: ignore

            if not isinstance(log_probs, torch.Tensor):
                raise RuntimeError("Error while computing the distribution")

            # Computing difference between critics_evaluation and bootstrapped evaluation
            advantage = rewards_tensor + remaining_reward - critics_evaluation
            critics_loss = advantage.square()  # MSE for critic
            actors_loss = -advantage.detach() * log_probs.reshape(
                -1, 1
            )  # Policy gradient

            # Backpropagate combined loss (averaged out over batch)
            total_loss = (critics_loss + actors_loss).mean()
            total_loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

            # Clear buffer after update
            self.trajectory_buffer.clear()

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

        print(f"Saving model to {path}...")
        os.makedirs(path, exist_ok=True)

        config: Dict[str, Any] = {
            "state_space_shape": self.state_space_shape,
            "action_space_size": self.action_space_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.model.to("cpu").state_dict(), os.path.join(path, "model.pth"))

    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> ActorCriticAgent:
        """
        Load a model from disk.
        Recreates the architecture using config.json and loads the trained weights.
        """
        print(f"Loading model from {path}...")

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        config["hidden_layer_sizes"] = tuple(config["hidden_layer_sizes"])
        agent = cls(device=device, **config)

        weights_path = os.path.join(path, "model.pth")
        agent.model.load_state_dict(
            torch.load(weights_path, map_location=torch.device(device))
        )

        print("Model loaded successfully.")
        return agent
