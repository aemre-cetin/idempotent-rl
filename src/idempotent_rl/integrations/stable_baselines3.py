import torch
from ..compactor import InplacePERCompactor

class SB3PrioritizedReplayBufferInplaceHook:
    """
    Zero-copy in-place compaction hook for Stable-Baselines3 Prioritized Replay Buffers.
    Reorganizes transitions on GPU without host cudaMalloc calls, preserving physical contiguity.
    """
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.compactor = InplacePERCompactor(state_dim=state_dim, action_dim=action_dim)
        self.capacity = capacity

    def __call__(self, buffer_observations, buffer_actions, buffer_rewards, buffer_next_observations, buffer_dones, priorities):
        return self.compactor.compact(
            buffer_observations, buffer_actions, buffer_rewards,
            buffer_next_observations, buffer_dones,
            priorities, capacity=self.capacity
        )
