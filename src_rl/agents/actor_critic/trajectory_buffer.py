from typing import NamedTuple, List, Tuple, Any, Callable
import numpy as np
from operator import attrgetter


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class TrajectoryBuffer:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.buffer: List[Transition] = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_len

    def clear(self):
        self.buffer.clear()

    def collect_array(self, attr: Callable[[Transition], Any], dtype: np.dtype[Any]):
        collected = np.array([attr(t) for t in self.buffer], dtype=dtype)
        return collected

    def get_batch(self) -> Tuple[np.ndarray, ...]:
        states = self.collect_array(attrgetter("state"), np.dtype(np.float32))
        actions = self.collect_array(attrgetter("action"), np.dtype(np.int32))
        rewards = self.collect_array(attrgetter("reward"), np.dtype(np.float32))
        next_states = self.collect_array(attrgetter("next_state"), np.dtype(np.float32))
        dones = self.collect_array(attrgetter("done"), np.dtype(np.bool))
        return states, actions, rewards, next_states, dones
