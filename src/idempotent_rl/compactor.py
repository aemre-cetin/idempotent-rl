import torch
from .kernel import compact_replay_buffer_inplace

def generate_idempotent_per_map(td_errors, capacity, device=None):
    """
    Constructs an idempotent permutation map f(x) for Prioritized Experience Replay.
    Top-K transitions with highest TD-error are consolidated into contiguous active indices [0, capacity - 1].
    """
    if device is None:
        device = td_errors.device

    B, N = td_errors.shape
    target_map = torch.arange(N, dtype=torch.int32, device=device).unsqueeze(0).expand(B, N).clone()

    for b in range(B):
        sorted_indices = torch.argsort(td_errors[b], descending=True)
        top_k_indices = sorted_indices[:capacity]

        active_tail = top_k_indices[top_k_indices >= capacity]
        num_swaps = active_tail.numel()

        if num_swaps > 0:
            is_in_top_k = torch.zeros(capacity, dtype=torch.bool, device=device)
            head_in_top_k = top_k_indices[top_k_indices < capacity]
            is_in_top_k[head_in_top_k] = True

            vacant_head = torch.nonzero(~is_in_top_k, as_tuple=True)[0][:num_swaps]

            target_map[b, vacant_head] = active_tail.to(torch.int32)
            target_map[b, active_tail] = vacant_head.to(torch.int32)

    return target_map


class InplacePERCompactor:
    """
    High-level Zero-Copy Prioritized Experience Replay (PER) Compactor.
    Consolidates transitions directly in accelerator VRAM without secondary buffer allocations.
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def compact(self, states, actions, rewards, next_states, dones, td_errors, capacity: int):
        """
        Executes in-place consolidation of the transition tuple using TD-error priorities.
        Returns sliced active buffers [:, :capacity, ...] contiguous in memory.
        """
        device = states.device
        target_map = generate_idempotent_per_map(td_errors, capacity, device=device)

        compact_replay_buffer_inplace(states, actions, rewards, next_states, dones, target_map)

        return (
            states[:, :capacity, :],
            actions[:, :capacity, :],
            rewards[:, :capacity],
            next_states[:, :capacity, :],
            dones[:, :capacity]
        )

    def __call__(self, *args, **kwargs):
        return self.compact(*args, **kwargs)
