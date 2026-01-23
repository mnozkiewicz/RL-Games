from .base_agent import BaseAgent
from .policy_gradient.actor_critic import ActorCriticAgent
from .policy_gradient.ppo import PPOAgent
from typing import Literal, Tuple, Union


def create_agent(
    agent_name: str,
    state_space_shape: Union[Tuple[int, int, int], int],
    action_space_size: int,
    input_type: Literal["processed_state", "raw_pixels"],
) -> BaseAgent:
    if agent_name == "ppo":
        return PPOAgent(
            state_space_shape=state_space_shape,
            action_space_size=action_space_size,
            input_type=input_type,
        )
    elif agent_name == "actor_critic":
        return ActorCriticAgent(
            state_space_shape=state_space_shape,
            action_space_size=action_space_size,
            input_type=input_type,
        )
    else:
        raise ValueError(f"Unknown agent algorithm {agent_name}")


def load_pretrained_model(agent_name: str, path: str, device: str) -> BaseAgent:
    if agent_name == "ppo":
        return PPOAgent.load_model(path=path, device=device)
    elif agent_name == "actor_critic":
        return ActorCriticAgent.load_model(path=path, device=device)
    else:
        raise ValueError(f"Unknown agent algorithm {agent_name}")
