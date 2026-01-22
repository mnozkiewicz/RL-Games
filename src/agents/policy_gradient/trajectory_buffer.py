from typing import NamedTuple, List, Any, Callable
import numpy as np
from operator import attrgetter


class Transition(NamedTuple):
    """
    Represents a single step (transition) in the environment.
    Each transition contains:
    - state: the observation at time t
    - action: the action taken at time t
    - reward: the reward received after taking the action
    - next_state: the observation at time t+1
    - done: whether the episode ended after this step
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray


class TrajectoryBuffer:
    """
    Fixed-size buffer to store transitions for an actor-critic agent.
    Used for batching updates to the networks.
    """

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.buffer: List[Transition] = []  # Stores Transition namedtuples

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
    ) -> None:
        """
        Add a new transition to the buffer.
        If the buffer exceeds max_len, discard the oldest transition.
        """
        self.buffer.append(
            Transition(state, action, reward, next_state, done, log_prob)
        )
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        """
        return len(self.buffer) >= self.max_len

    def clear(self) -> None:
        """
        Empty the buffer.
        Called after an update to prepare for the next batch.
        """
        self.buffer.clear()

    def collect_array(
        self, attr: Callable[[Transition], Any], dtype: np.dtype[Any]
    ) -> np.ndarray:
        """
        Helper function to extract a specific attribute from all transitions
        and return it as a NumPy array of the specified dtype.
        """
        collected = np.array([attr(t) for t in self.buffer], dtype=dtype)
        return collected

    def get_batch(self) -> Batch:
        """
        Convert the stored transitions into arrays suitable for training.
        Returns a tuple: (states, actions, rewards, next_states, dones)
        Where all are arrays.
        """
        states = self.collect_array(attrgetter("state"), np.dtype(np.float32))
        actions = self.collect_array(attrgetter("action"), np.dtype(np.int32))
        rewards = self.collect_array(attrgetter("reward"), np.dtype(np.float32))
        next_states = self.collect_array(attrgetter("next_state"), np.dtype(np.float32))
        dones = self.collect_array(attrgetter("done"), np.dtype(np.bool))
        log_probs = self.collect_array(attrgetter("log_prob"), np.dtype(np.float32))

        return Batch(states, actions, rewards, next_states, dones, log_probs)
