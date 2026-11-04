"""
KubeVRAM: AI-Assisted Kubernetes Inference Bridge & Day-2 GPU Operations CLI.
"""

from kubevram.collectors import (
    CollectorError,
    collect_live_cluster_state,
    load_cluster_from_file,
    resolve_cluster_state,
)
from kubevram.oom_correlator import correlate_pod_failure, triage_cluster
from kubevram.partition_advisor import is_mig_supported, recommend_partitioning
from kubevram.rbac_auditor import audit_cluster_rbac
from kubevram.vllm_calculator import GPU_CATALOG, MODEL_CATALOG, calculate_vllm_tuning

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "CollectorError",
    "collect_live_cluster_state",
    "load_cluster_from_file",
    "resolve_cluster_state",
    "correlate_pod_failure",
    "triage_cluster",
    "audit_cluster_rbac",
    "calculate_vllm_tuning",
    "recommend_partitioning",
    "is_mig_supported",
    "MODEL_CATALOG",
    "GPU_CATALOG",
]
