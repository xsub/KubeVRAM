"""
KubeVRAM GPU Partitioning & Sharing Strategy Advisor.

Evaluates hardware capabilities (MIG vs non-MIG), workload memory characteristics,
and tenancy requirements to recommend:
1. Multi-Instance GPU (MIG) silicon partitioning.
2. CUDA Multi-Process Service (MPS) concurrent kernel execution.
3. GPU Time-Slicing temporal sharing.
4. Dedicated Full-GPU allocation.

Generates ready-to-apply ConfigMaps for the NVIDIA GPU Operator (mig-parted and device-plugin).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

# Hardware architecture capabilities
MIG_CAPABLE_GPU_KEYWORDS: Set[str] = {
    "H100",
    "A100",
    "A30",
    "H200",
    "B100",
    "B200",
}

NON_MIG_GPU_KEYWORDS: Set[str] = {
    "L40S",
    "L40",
    "A10G",
    "A10",
    "T4",
    "L4",
    "V100",
    "RTX",
}

# Standard MIG slice capacities and slice counts
MIG_80GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.10gb": {"vram_gb": 10.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.20gb": {"vram_gb": 20.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.40gb": {"vram_gb": 40.0, "max_slices": 2, "sm_fraction": "3/7"},
    "4g.40gb": {"vram_gb": 40.0, "max_slices": 1, "sm_fraction": "4/7"},
    "7g.80gb": {"vram_gb": 80.0, "max_slices": 1, "sm_fraction": "7/7"},
}

MIG_40GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.5gb": {"vram_gb": 5.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.10gb": {"vram_gb": 10.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.20gb": {"vram_gb": 20.0, "max_slices": 2, "sm_fraction": "3/7"},
    "7g.40gb": {"vram_gb": 40.0, "max_slices": 1, "sm_fraction": "7/7"},
}


@dataclass
class PartitionRecommendation:
    """Structured recommendation for GPU partitioning strategy and operator configuration."""

    gpu_sku: str
    supports_mig: bool
    workload_type: str
    tenancy: str
    model_vram_gb: float
    recommended_strategy: str  # "MIG" | "MPS" | "Time-Slicing" | "Full-GPU"
    mig_profile: Optional[str]
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    rationale: str
    warnings: List[str] = field(default_factory=list)
    configmap_yaml: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return asdict(self)


def is_mig_supported(gpu_sku: str) -> bool:
    """
    Evaluates whether the given GPU hardware SKU supports NVIDIA Multi-Instance GPU (MIG).
    H100, A100, A30, H200 support MIG.
    L40S, A10G, T4, L4 do NOT.
    """
    sku_upper = gpu_sku.upper().strip()

    # 1. Check MIG-capable families first
    for mig in MIG_CAPABLE_GPU_KEYWORDS:
        if mig in sku_upper:
            return True

    # 2. If it's an explicit MIG slice
    if sku_upper.startswith("MIG-") or ".GB" in sku_upper:
        return True

    # 3. Explicit non-MIG families or unknown
    return False


def _select_mig_profile(gpu_sku: str, model_vram_gb: float) -> Optional[str]:
    """Selects the most suitable MIG profile based on SKU capacity and required model VRAM."""
    sku_upper = gpu_sku.upper()

    if "40GB" in sku_upper and "A100" in sku_upper:
        profiles = MIG_40GB_PROFILES
    else:
        profiles = MIG_80GB_PROFILES

    # Find the smallest slice that accommodates model_vram_gb with headroom
    for profile, info in sorted(profiles.items(), key=lambda x: x[1]["vram_gb"]):
        if model_vram_gb <= info["vram_gb"]:
            return profile

    return None


def _generate_mig_configmap(mig_profile: str, gpu_sku: str) -> str:
    """Generates mig-parted ConfigMap for the NVIDIA GPU Operator."""
    sku_upper = gpu_sku.upper()
    if "40GB" in sku_upper and "A100" in sku_upper:
        slices = MIG_40GB_PROFILES.get(mig_profile, {}).get("max_slices", 1)
    else:
        slices = MIG_80GB_PROFILES.get(mig_profile, {}).get("max_slices", 1)

    profile_name = f"profile-{mig_profile.replace('.', '-')}"

    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-mig-parted-configs
  namespace: gpu-operator
data:
  config.yaml: |-
    version: v1
    mig-configs:
      {profile_name}:
        - devices: all
          mig-enabled: true
          mig-devices:
            "{mig_profile}": {slices}
"""


def _generate_time_slicing_configmap(replicas: int = 4) -> str:
    """Generates time-slicing ConfigMap for the NVIDIA GPU Operator."""
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-sharing-configs
  namespace: gpu-operator
data:
  default: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        renameByDefault: false
        failRequestsGreaterThanOne: true
        resources:
          - name: nvidia.com/gpu
            replicas: {replicas}
"""


def _generate_mps_configmap(replicas: int = 2) -> str:
    """Generates CUDA MPS ConfigMap for the NVIDIA GPU Operator."""
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: mps-sharing-config
  namespace: gpu-operator
data:
  mps-profile: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      mps:
        renameByDefault: false
        failRequestsGreaterThanOne: true
        resources:
          - name: nvidia.com/gpu
            replicas: {replicas}
"""


def _generate_full_gpu_configmap() -> str:
    """Generates default unshared Full-GPU ConfigMap for the NVIDIA GPU Operator."""
    return """apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-sharing-configs
  namespace: gpu-operator
data:
  default: |-
    version: v1
    flags:
      migStrategy: none
"""


def recommend_partitioning(
    gpu_sku: str = "H100-80GB",
    workload_type: str = "vllm",
    tenancy: str = "multi",
    model_vram_gb: float = 16.0,
) -> Dict[str, Any]:
    """
    Recommends optimal GPU sharing mechanism (MIG, MPS, Time-Slicing, or Full-GPU)
    based on hardware capabilities, workload type, tenancy model, and VRAM requirements.
    """
    supports_mig = is_mig_supported(gpu_sku)
    norm_workload = workload_type.strip().lower()
    norm_tenancy = tenancy.strip().lower()
    model_vram_gb = float(model_vram_gb)

    warnings: List[str] = []
    recommended_strategy: str = "Full-GPU"
    mig_profile: Optional[str] = None
    risk_level: str = "LOW"
    rationale: str = ""
    configmap_yaml: str = ""

    # Universal check: vLLM workloads vs Time-Slicing
    if norm_workload == "vllm":
        warnings.append(
            "CRITICAL: Do NOT use Time-Slicing for vLLM! vLLM pre-allocates visible physical VRAM "
            "(--gpu-memory-utilization, default 0.90) for weights and KV-cache. Under Time-Slicing, "
            "all co-located pods view the entire physical VRAM, causing secondary pods to crash "
            "immediately with torch.cuda.OutOfMemoryError (Exit Code 1)."
        )

    # 1. Dev Notebooks / Interactive Sandboxes
    if norm_workload == "dev-notebook":
        recommended_strategy = "Time-Slicing"
        mig_profile = None
        risk_level = "LOW"
        rationale = (
            "Interactive development notebooks and sandboxes feature bursty, idle-heavy access patterns "
            "(>90% idle time). Time-Slicing with temporal context switching (replicas: 4) maximizes "
            "developer density without reserving static silicon."
        )
        warnings.append(
            "Time-Slicing does not enforce memory boundaries between containers. Ensure developers "
            "apply framework-level memory limits to avoid mutual OOM eviction."
        )
        configmap_yaml = _generate_time_slicing_configmap(replicas=4)

    # 2. Large Frontier Models (> 40GB)
    elif model_vram_gb > 40.0:
        recommended_strategy = "Full-GPU"
        mig_profile = None
        risk_level = "LOW"
        rationale = (
            f"Model requires {model_vram_gb:.1f} GB VRAM, consuming the majority of GPU capacity. "
            "Dedicated Full-GPU allocation ensures 100% memory bandwidth, full SM access, and enables "
            "NVLink P2P communication for Tensor Parallelism."
        )
        configmap_yaml = _generate_full_gpu_configmap()

    # 3. vLLM Serving Workloads
    elif norm_workload == "vllm":
        if norm_tenancy == "multi":
            if supports_mig:
                mig_profile = _select_mig_profile(gpu_sku, model_vram_gb)
                if mig_profile and mig_profile != "7g.80gb":
                    recommended_strategy = "MIG"
                    risk_level = "LOW"
                    rationale = (
                        f"Hardware MIG profile '{mig_profile}' provides dedicated hardware silicon partitions "
                        "(isolated SMs, memory controllers, and L2 cache). This guarantees predictable "
                        "vLLM PagedAttention allocation and completely prevents co-tenant memory collisions."
                    )
                    configmap_yaml = _generate_mig_configmap(mig_profile, gpu_sku)
                else:
                    recommended_strategy = "Full-GPU"
                    mig_profile = None
                    risk_level = "LOW"
                    rationale = (
                        "Model memory requirement approaches full device capacity. Dedicated Full-GPU "
                        "allocation provides optimal throughput and avoids MIG reconfiguration overhead."
                    )
                    configmap_yaml = _generate_full_gpu_configmap()
            else:
                recommended_strategy = "Full-GPU"
                mig_profile = None
                risk_level = "HIGH"
                rationale = (
                    f"GPU SKU '{gpu_sku}' does not support hardware MIG partitioning. Because vLLM eagerly "
                    "pre-allocates memory, Time-Slicing causes immediate OOM crashes. Dedicated Full-GPU "
                    "allocation per replica is strongly recommended to guarantee production stability."
                )
                warnings.append(
                    f"GPU '{gpu_sku}' lacks silicon MIG support. Multi-tenant vLLM on this device is high risk; "
                    "dedicate 1 physical GPU per vLLM container."
                )
                configmap_yaml = _generate_full_gpu_configmap()
        else:  # Single-tenancy vLLM
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = (
                "Dedicated single-tenant vLLM serving. Full-GPU allocation ensures maximum throughput, "
                "lowest Time-To-First-Token (TTFT), and full KV-cache headroom."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 4. Auxiliary Microservices: Embeddings & RAG Pipelines
    elif norm_workload in ("embeddings", "rag-pipeline"):
        if norm_tenancy == "multi":
            if supports_mig:
                mig_profile = _select_mig_profile(gpu_sku, model_vram_gb) or "1g.10gb"
                recommended_strategy = "MIG"
                risk_level = "LOW"
                rationale = (
                    f"MIG profile '{mig_profile}' isolates auxiliary embedding and reranking microservices "
                    "with dedicated SMs and HBM, ensuring steady p99 latencies without interference."
                )
                configmap_yaml = _generate_mig_configmap(mig_profile, gpu_sku)
            else:
                recommended_strategy = "MPS"
                mig_profile = None
                risk_level = "LOW"
                rationale = (
                    "GPU does not support MIG. CUDA Multi-Process Service (MPS) funnels small embedding "
                    "and reranker kernels into a single CUDA context, eliminating context-switch latency "
                    "and enabling concurrent SM execution."
                )
                warnings.append(
                    "CUDA MPS shares a unified CUDA address space: an unhandled memory exception in one "
                    "container will reset the shared GPU context across all co-located MPS workers."
                )
                configmap_yaml = _generate_mps_configmap(replicas=2)
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = (
                "Single-tenant dedicated deployment for embedding/RAG pipeline. Provides uncontended SM "
                "and memory access."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 5. Triton Inference Server Workloads
    elif norm_workload == "triton":
        if norm_tenancy == "multi":
            if supports_mig:
                mig_profile = _select_mig_profile(gpu_sku, model_vram_gb) or "2g.20gb"
                recommended_strategy = "MIG"
                risk_level = "LOW"
                rationale = (
                    f"Hardware MIG partition '{mig_profile}' isolates Triton model execution instances, "
                    "preventing noisy-neighbor compute degradation and memory over-subscription."
                )
                configmap_yaml = _generate_mig_configmap(mig_profile, gpu_sku)
            else:
                recommended_strategy = "MPS"
                mig_profile = None
                risk_level = "MEDIUM"
                rationale = (
                    "Hardware lacks MIG. CUDA MPS allows multiple Triton model instances to overlap kernel "
                    "execution concurrently."
                )
                warnings.append(
                    "Error containment limitation: CUDA MPS lacks hardware memory fault isolation."
                )
                configmap_yaml = _generate_mps_configmap(replicas=2)
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = (
                "Dedicated Full-GPU allocation allows Triton to utilize dynamic batching across 100% of "
                "available SMs and GPU memory."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 6. Fallback / Generic Workload
    else:
        if supports_mig and norm_tenancy == "multi" and model_vram_gb <= 20.0:
            mig_profile = _select_mig_profile(gpu_sku, model_vram_gb) or "2g.20gb"
            recommended_strategy = "MIG"
            risk_level = "LOW"
            rationale = f"MIG partitioning ({mig_profile}) provides hardware isolation for multi-tenant serving."
            configmap_yaml = _generate_mig_configmap(mig_profile, gpu_sku)
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = "Dedicated Full-GPU allocation ensures maximum stability and performance."
            configmap_yaml = _generate_full_gpu_configmap()

    rec = PartitionRecommendation(
        gpu_sku=gpu_sku,
        supports_mig=supports_mig,
        workload_type=norm_workload,
        tenancy=norm_tenancy,
        model_vram_gb=model_vram_gb,
        recommended_strategy=recommended_strategy,
        mig_profile=mig_profile,
        risk_level=risk_level,
        rationale=rationale,
        warnings=warnings,
        configmap_yaml=configmap_yaml,
    )
    return rec.to_dict()


if __name__ == "__main__":
    import json
    import sys

    # Quick test execution
    res = recommend_partitioning(
        gpu_sku="H100-80GB",
        workload_type="vllm",
        tenancy="multi",
        model_vram_gb=16.0,
    )
    print(json.dumps(res, indent=2))
