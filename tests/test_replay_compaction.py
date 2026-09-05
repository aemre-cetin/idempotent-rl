import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from idempotent_rl import InplacePERCompactor, generate_idempotent_per_map, compact_replay_buffer_inplace

def test_rl_replay_compaction():
    assert torch.cuda.is_available(), "CUDA required for GPU testing"
    device = torch.device("cuda:0")

    NUM_ENVS = 16         # 16 Parallel environments
    TRANS_PER_ENV = 2048  # 2,048 transitions per environment
    STATE_DIM = 64        # State feature dimension
    ACTION_DIM = 16       # Action feature dimension
    CAPACITY = 1024       # 50% Compaction

    print(f"Testing idempotent-rl on {torch.cuda.get_device_name(0)}...")
    print(f"Shape: Envs={NUM_ENVS}, Trans={TRANS_PER_ENV}, S_Dim={STATE_DIM}, A_Dim={ACTION_DIM}, Capacity={CAPACITY}")

    # Allocate Transition Tensors (Float32)
    states = torch.randn((NUM_ENVS, TRANS_PER_ENV, STATE_DIM), dtype=torch.float32, device=device)
    actions = torch.randn((NUM_ENVS, TRANS_PER_ENV, ACTION_DIM), dtype=torch.float32, device=device)
    rewards = torch.randn((NUM_ENVS, TRANS_PER_ENV), dtype=torch.float32, device=device)
    next_states = torch.randn((NUM_ENVS, TRANS_PER_ENV, STATE_DIM), dtype=torch.float32, device=device)
    dones = torch.randint(0, 2, (NUM_ENVS, TRANS_PER_ENV), dtype=torch.float32, device=device)

    td_errors = torch.rand((NUM_ENVS, TRANS_PER_ENV), dtype=torch.float32, device=device)
    target_map = generate_idempotent_per_map(td_errors, CAPACITY, device=device)

    # Reference Out-of-Place Gather
    s_ref = torch.gather(states, 1, target_map.unsqueeze(-1).expand_as(states))[:, :CAPACITY, :].clone()
    a_ref = torch.gather(actions, 1, target_map.unsqueeze(-1).expand_as(actions))[:, :CAPACITY, :].clone()
    r_ref = torch.gather(rewards, 1, target_map)[:, :CAPACITY].clone()
    ns_ref = torch.gather(next_states, 1, target_map.unsqueeze(-1).expand_as(next_states))[:, :CAPACITY, :].clone()
    d_ref = torch.gather(dones, 1, target_map)[:, :CAPACITY].clone()

    # In-Situ Compaction via InplacePERCompactor
    compactor = InplacePERCompactor(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    s_test, a_test, r_test, ns_test, d_test = compactor(
        states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone(),
        td_errors, capacity=CAPACITY
    )

    # In-Situ Kernel Zero-Memory Verification
    s_work = states.clone()
    a_work = actions.clone()
    r_work = rewards.clone()
    ns_work = next_states.clone()
    d_work = dones.clone()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    compact_replay_buffer_inplace(s_work, a_work, r_work, ns_work, d_work, target_map)

    mem_after = torch.cuda.max_memory_allocated(device)
    aux_memory = mem_after - mem_before

    diff_s = torch.max(torch.abs(s_ref - s_test)).item()
    diff_a = torch.max(torch.abs(a_ref - a_test)).item()
    diff_r = torch.max(torch.abs(r_ref - r_test)).item()
    diff_ns = torch.max(torch.abs(ns_ref - ns_test)).item()
    diff_d = torch.max(torch.abs(d_ref - d_test)).item()

    max_diff = max(diff_s, diff_a, diff_r, diff_ns, diff_d)

    nan_count = (
        torch.isnan(s_test).sum().item() +
        torch.isnan(a_test).sum().item() +
        torch.isnan(r_test).sum().item() +
        torch.isnan(ns_test).sum().item() +
        torch.isnan(d_test).sum().item()
    )

    print(f"Kernel Auxiliary Memory Allocated: {aux_memory} bytes")
    print(f"Maximum Multi-Tensor Parity Difference: {max_diff:.6f}")
    print(f"Total NaN Detections: {nan_count}")

    assert aux_memory == 0, f"Expected 0 bytes auxiliary memory, got {aux_memory}"
    assert max_diff == 0.0, f"Expected exact numerical parity, got diff={max_diff}"
    assert nan_count == 0, "NaN detected in compacted replay buffer output"

    print("[SUCCESS] Test passed! In-situ RL multi-tensor compaction is 100% Zero-Copy, Bit-exact and NaN-free.")

if __name__ == "__main__":
    test_rl_replay_compaction()

