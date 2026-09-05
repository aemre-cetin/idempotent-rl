import torch
import triton
import triton.language as tl

@triton.jit
def _inplace_rl_per_compact_kernel(
    States_ptr,            # [B, N, S_DIM]
    Actions_ptr,           # [B, N, A_DIM]
    Rewards_ptr,           # [B, N]
    NextStates_ptr,        # [B, N, S_DIM]
    Dones_ptr,             # [B, N]
    TargetMap_ptr,         # [B, N]
    # Strides for States
    stride_sb, stride_sn, stride_sd,
    # Strides for Actions
    stride_ab, stride_an, stride_ad,
    # Strides for Rewards
    stride_rb, stride_rn,
    # Strides for NextStates
    stride_nsb, stride_nsn, stride_nsd,
    # Strides for Dones
    stride_db, stride_dn,
    # Strides for TargetMap
    stride_mb, stride_mn,
    # Dimensions
    N: tl.constexpr,       # Number of transitions per partition
    S_DIM: tl.constexpr,   # State feature dimension (e.g. 64)
    A_DIM: tl.constexpr    # Action feature dimension (e.g. 8 or 16)
):
    pid_batch = tl.program_id(0)

    # Base pointers for the current partition/environment
    s_base_ptr   = States_ptr     + pid_batch * stride_sb
    a_base_ptr   = Actions_ptr    + pid_batch * stride_ab
    r_base_ptr   = Rewards_ptr    + pid_batch * stride_rb
    ns_base_ptr  = NextStates_ptr + pid_batch * stride_nsb
    d_base_ptr   = Dones_ptr      + pid_batch * stride_db
    map_base_ptr = TargetMap_ptr  + pid_batch * stride_mb

    offs_s = tl.arange(0, S_DIM)
    offs_a = tl.arange(0, A_DIM)

    # Traversal across N transitions
    for i in range(0, N):
        dest_idx = tl.load(map_base_ptr + i * stride_mn)

        # Leader Condition: i can only be a cycle leader if dest_idx > i
        # (If dest_idx == i: fixed point; if dest_idx < i: already handled by an earlier index)
        if dest_idx > i:
            second_hop = tl.load(map_base_ptr + dest_idx * stride_mn)

            if second_hop == i:
                # -------------------------------------------------------------
                # 2-Cycle (Transposition) Fast-Path: Direct in-register swap
                # -------------------------------------------------------------
                # 1) States [S_DIM]
                ptr_s_i = s_base_ptr + i * stride_sn + offs_s * stride_sd
                ptr_s_d = s_base_ptr + dest_idx * stride_sn + offs_s * stride_sd
                val_s_i = tl.load(ptr_s_i)
                val_s_d = tl.load(ptr_s_d)
                tl.store(ptr_s_i, val_s_d)
                tl.store(ptr_s_d, val_s_i)

                # 2) Next States [S_DIM]
                ptr_ns_i = ns_base_ptr + i * stride_nsn + offs_s * stride_nsd
                ptr_ns_d = ns_base_ptr + dest_idx * stride_nsn + offs_s * stride_nsd
                val_ns_i = tl.load(ptr_ns_i)
                val_ns_d = tl.load(ptr_ns_d)
                tl.store(ptr_ns_i, val_ns_d)
                tl.store(ptr_ns_d, val_ns_i)

                # 3) Actions [A_DIM]
                ptr_a_i = a_base_ptr + i * stride_an + offs_a * stride_ad
                ptr_a_d = a_base_ptr + dest_idx * stride_an + offs_a * stride_ad
                val_a_i = tl.load(ptr_a_i)
                val_a_d = tl.load(ptr_a_d)
                tl.store(ptr_a_i, val_a_d)
                tl.store(ptr_a_d, val_a_i)

                # 4) Rewards [Scalar]
                ptr_r_i = r_base_ptr + i * stride_rn
                ptr_r_d = r_base_ptr + dest_idx * stride_rn
                val_r_i = tl.load(ptr_r_i)
                val_r_d = tl.load(ptr_r_d)
                tl.store(ptr_r_i, val_r_d)
                tl.store(ptr_r_d, val_r_i)

                # 5) Dones [Scalar]
                ptr_d_i = d_base_ptr + i * stride_dn
                ptr_d_d = d_base_ptr + dest_idx * stride_dn
                val_d_i = tl.load(ptr_d_i)
                val_d_d = tl.load(ptr_d_d)
                tl.store(ptr_d_i, val_d_d)
                tl.store(ptr_d_d, val_d_i)

            else:
                # -------------------------------------------------------------
                # Generalized Disjoint Cycle Traversal (Cycle Length L >= 3)
                # -------------------------------------------------------------
                curr = dest_idx
                is_leader = True
                keep_searching = True

                while keep_searching:
                    if curr == i:
                        keep_searching = False
                    else:
                        if curr < i:
                            is_leader = False
                            keep_searching = False
                        else:
                            curr = tl.load(map_base_ptr + curr * stride_mn)

                if is_leader:
                    # Hold head transition in temporary registers
                    temp_s  = tl.load(s_base_ptr  + i * stride_sn  + offs_s * stride_sd)
                    temp_ns = tl.load(ns_base_ptr + i * stride_nsn + offs_s * stride_nsd)
                    temp_a  = tl.load(a_base_ptr  + i * stride_an  + offs_a * stride_ad)
                    temp_r  = tl.load(r_base_ptr  + i * stride_rn)
                    temp_d  = tl.load(d_base_ptr  + i * stride_dn)

                    curr_slot = i
                    cycle_active = True

                    while cycle_active:
                        next_slot = tl.load(map_base_ptr + curr_slot * stride_mn)

                        if next_slot == i:
                            tl.store(s_base_ptr  + curr_slot * stride_sn  + offs_s * stride_sd, temp_s)
                            tl.store(ns_base_ptr + curr_slot * stride_nsn + offs_s * stride_nsd, temp_ns)
                            tl.store(a_base_ptr  + curr_slot * stride_an  + offs_a * stride_ad, temp_a)
                            tl.store(r_base_ptr  + curr_slot * stride_rn, temp_r)
                            tl.store(d_base_ptr  + curr_slot * stride_dn, temp_d)
                            cycle_active = False
                        else:
                            from_s  = tl.load(s_base_ptr  + next_slot * stride_sn  + offs_s * stride_sd)
                            from_ns = tl.load(ns_base_ptr + next_slot * stride_nsn + offs_s * stride_nsd)
                            from_a  = tl.load(a_base_ptr  + next_slot * stride_an  + offs_a * stride_ad)
                            from_r  = tl.load(r_base_ptr  + next_slot * stride_rn)
                            from_d  = tl.load(d_base_ptr  + next_slot * stride_dn)

                            tl.store(s_base_ptr  + curr_slot * stride_sn  + offs_s * stride_sd, from_s)
                            tl.store(ns_base_ptr + curr_slot * stride_nsn + offs_s * stride_nsd, from_ns)
                            tl.store(a_base_ptr  + curr_slot * stride_an  + offs_a * stride_ad, from_a)
                            tl.store(r_base_ptr  + curr_slot * stride_rn, from_r)
                            tl.store(d_base_ptr  + curr_slot * stride_dn, from_d)

                            curr_slot = next_slot


def compact_replay_buffer_inplace(states, actions, rewards, next_states, dones, target_map):
    """
    In-Place Zero-Copy Prioritized Experience Replay (PER) Compactor.
    Permutes heterogeneous transition tensors in-place using O(1) auxiliary scalar memory.
    """
    assert states.is_contiguous(), "States must be contiguous"
    assert actions.is_contiguous(), "Actions must be contiguous"
    assert rewards.is_contiguous(), "Rewards must be contiguous"
    assert next_states.is_contiguous(), "NextStates must be contiguous"
    assert dones.is_contiguous(), "Dones must be contiguous"
    assert target_map.is_contiguous(), "TargetMap must be contiguous"

    B, N, S_DIM = states.shape
    _, _, A_DIM = actions.shape

    grid = (B,)

    _inplace_rl_per_compact_kernel[grid](
        states, actions, rewards, next_states, dones, target_map,
        states.stride(0), states.stride(1), states.stride(2),
        actions.stride(0), actions.stride(1), actions.stride(2),
        rewards.stride(0), rewards.stride(1),
        next_states.stride(0), next_states.stride(1), next_states.stride(2),
        dones.stride(0), dones.stride(1),
        target_map.stride(0), target_map.stride(1),
        N=N,
        S_DIM=S_DIM,
        A_DIM=A_DIM
    )
