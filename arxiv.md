# arXiv & ResearchGate Submission Metadata: Pillar 2 (idempotent-rl)

## 1. arXiv Metadata

- **Title:**
  Zero-Copy In-Place Prioritized Experience Replay via Idempotent Disjoint Cycles in Deep Reinforcement Learning Accelerators

- **Authors:**
  Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)

- **Primary Category:**
  `cs.LG` (Machine Learning)

- **Secondary Categories:**
  `cs.AI` (Artificial Intelligence), `cs.RO` (Robotics), `cs.DC` (Distributed, Computing, and Cluster Computing)

- **Comments:**
  4 pages, 3 figures. Reference implementation and Triton kernels available at https://github.com/aemre-cetin/idempotent-rl. Protected under U.S. Patent Application No. 64/148,668.

- **ACM Classification:**
  I.2.6; I.2.8; C.1.4

- **MSC Classification:**
  68T05; 68W10

- **Archive Package:**
  `paper/arxiv_package_idempotent_rl.tar.gz`

### Abstract (Formatted for arXiv Form):
Deep Reinforcement Learning (RL) architectures, such as Deep Q-Networks (DQN), Soft Actor-Critic (SAC), and Proximal Policy Optimization (PPO), rely critically on Prioritized Experience Replay (PER) to break temporal sample correlations and prioritize transitions exhibiting high Temporal Difference (TD) errors. However, as parallel simulation environments scale to millions of continuous transitions on modern accelerators (e.g., Isaac Gym, Brax), conventional systems suffer from severe hardware bottlenecks: they either introduce host-device PCIe bus stalls via host-managed tree structures (Sum-Trees) or induce catastrophic GPU memory allocation spikes (O(N * D) auxiliary buffers via operating system calls like cudaMalloc) during experience consolidation and eviction. In this paper, we propose a novel hardware-software co-designed architecture and high-performance GPU kernel for zero-copy in-place Prioritized Experience Replay compaction. By formulating transition prioritization as an algebraic mapping satisfying the idempotence condition (f(f(x)) = f(x)), our method stabilizes retained high-priority experiences into invariant fixed points and resolves non-contiguous transition rearrangements through mutually disjoint permutation cycles. We demonstrate that cycle leaders across heterogeneous multi-tensor fields (states, actions, rewards, next-states, episode terminations) can be verified on-the-fly with strictly O(1) scalar auxiliary register storage, completely eliminating marking bitmasks and temporary global buffers. We implement our algorithm as an optimized Triton kernel and benchmark it on an enterprise NVIDIA Blackwell GPU accelerator (sm_120) across 262,144 multi-dimensional transitions. Empirical results demonstrate a 100% elimination of peak auxiliary VRAM (dropping from 136.50 MB to exactly 0.00 MB), a high-throughput execution rate of 67.97 Million transitions/second (3.86 ms latency), zero numerical degradation, and guaranteed physical memory contiguity for downstream Tensor Core gradient updates.

---

## 2. ResearchGate Submission Metadata

- **Title:**
  Zero-Copy In-Place Prioritized Experience Replay via Idempotent Disjoint Cycles in Deep Reinforcement Learning Accelerators

- **Publication Type:**
  Preprint / Research Article

- **Author & Affiliation:**
  Dr. A. Emre ÇETİN (Computational Systems and Cognitive Architectures, Izmir, Turkey)

- **Skills / Topics:**
  Reinforcement Learning, Experience Replay, Prioritized Experience Replay, GPU Acceleration, Isaac Gym, CleanRL, Stable-Baselines3, Triton, Zero-Copy

- **Patent Disclosure:**
  U.S. Patent Application No. 64/148,668 ("Patent Pending", Confirmation No. 5890).

- **Associated Links:**
  - GitHub Repository: https://github.com/aemre-cetin/idempotent-rl
  - PyPI Package: https://pypi.org/project/idempotent-rl/