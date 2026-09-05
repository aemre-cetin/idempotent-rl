"""
idempotent-rl: Zero-Copy In-Place Multi-Tensor Compactor for Deep Reinforcement Learning Replay Buffers.
Protected under U.S. Patent Application No. 64/148,668 ("Patent Pending").
Author: Dr. A. Emre ÇETİN <aemre.cetin@gmail.com>
"""

from .kernel import compact_replay_buffer_inplace
from .compactor import generate_idempotent_per_map, InplacePERCompactor

__version__ = "0.1.0"
__author__ = "Dr. A. Emre ÇETİN"
__patent__ = "U.S. Patent Application No. 64/148,668 (Patent Pending, Confirmation No. 5890)"

__all__ = [
    "compact_replay_buffer_inplace",
    "generate_idempotent_per_map",
    "InplacePERCompactor",
    "__version__",
    "__author__",
    "__patent__"
]

