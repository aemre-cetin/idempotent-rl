import torch
from ..compactor import InplacePERCompactor

class CleanRLReplayBufferInplaceHook:
    """
    Zero-copy in-place compaction hook for CleanRL multi-environment vectorized replay buffers.
    Consolidates transitions on-the-fly directly inside GPU buffers without secondary allocations.
    """
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.compactor = InplacePERCompactor(state_dim=state_dim, action_dim=action_dim)
        self.capacity = capacity

    def __call__(self, rb_states, rb_actions, rb_rewards, rb_next_states, rb_dones, td_errors):
        return self.compactor.compact(
            rb_states, rb_actions, rb_rewards, rb_next_states, rb_dones,
            td_errors, capacity=self.capacity
        )

