"""
KubeVRAM: AI-Assisted Kubernetes Inference Bridge & Day-2 GPU Operations CLI.
"""

from kubevram.partition_advisor import is_mig_supported, recommend_partitioning
from kubevram.vllm_calculator import GPU_CATALOG, MODEL_CATALOG, calculate_vllm_tuning

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "calculate_vllm_tuning",
    "recommend_partitioning",
    "is_mig_supported",
    "MODEL_CATALOG",
    "GPU_CATALOG",
]
