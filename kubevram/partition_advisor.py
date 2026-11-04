"""
KubeVRAM GPU Partitioning & Sharing Strategy Advisor.

Evaluates hardware capabilities (MIG vs non-MIG), per-family MIG profiles, device VRAM capacity,
NVIDIA k8s-device-plugin equal MPS memory division, and tenancy requirements to recommend:
1. Multi-Instance GPU (MIG) silicon partitioning.
2. CUDA Multi-Process Service (MPS) concurrent kernel execution (only when >=2 replicas fit per-client limit).
3. GPU Time-Slicing temporal sharing (only when burst working sets fit physical VRAM).
4. Dedicated Full-GPU allocation.

Generates ready-to-apply ConfigMaps for the NVIDIA GPU Operator (mig-parted and device-plugin).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Family-accurate NVIDIA MIG slice profiles
MIG_A30_24GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.6gb": {"vram_gb": 6.0, "max_slices": 4, "sm_fraction": "1/4"},
    "2g.12gb": {"vram_gb": 12.0, "max_slices": 2, "sm_fraction": "2/4"},
    "4g.24gb": {"vram_gb": 24.0, "max_slices": 1, "sm_fraction": "4/4"},
}

MIG_40GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.5gb": {"vram_gb": 5.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.10gb": {"vram_gb": 10.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.20gb": {"vram_gb": 20.0, "max_slices": 2, "sm_fraction": "3/7"},
    "7g.40gb": {"vram_gb": 40.0, "max_slices": 1, "sm_fraction": "7/7"},
}

MIG_80GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.10gb": {"vram_gb": 10.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.20gb": {"vram_gb": 20.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.40gb": {"vram_gb": 40.0, "max_slices": 2, "sm_fraction": "3/7"},
    "4g.40gb": {"vram_gb": 40.0, "max_slices": 1, "sm_fraction": "4/7"},
    "7g.80gb": {"vram_gb": 80.0, "max_slices": 1, "sm_fraction": "7/7"},
}

MIG_94GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.12gb": {"vram_gb": 12.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.24gb": {"vram_gb": 24.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.47gb": {"vram_gb": 47.0, "max_slices": 2, "sm_fraction": "3/7"},
    "4g.47gb": {"vram_gb": 47.0, "max_slices": 1, "sm_fraction": "4/7"},
    "7g.94gb": {"vram_gb": 94.0, "max_slices": 1, "sm_fraction": "7/7"},
}

MIG_H200_141GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.18gb": {"vram_gb": 18.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.35gb": {"vram_gb": 35.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.71gb": {"vram_gb": 71.0, "max_slices": 2, "sm_fraction": "3/7"},
    "4g.71gb": {"vram_gb": 71.0, "max_slices": 1, "sm_fraction": "4/7"},
    "7g.141gb": {"vram_gb": 141.0, "max_slices": 1, "sm_fraction": "7/7"},
}

MIG_B200_180GB_PROFILES: Dict[str, Dict[str, Any]] = {
    "1g.23gb": {"vram_gb": 23.0, "max_slices": 7, "sm_fraction": "1/7"},
    "2g.45gb": {"vram_gb": 45.0, "max_slices": 3, "sm_fraction": "2/7"},
    "3g.90gb": {"vram_gb": 90.0, "max_slices": 2, "sm_fraction": "3/7"},
    "4g.90gb": {"vram_gb": 90.0, "max_slices": 1, "sm_fraction": "4/7"},
    "7g.180gb": {"vram_gb": 180.0, "max_slices": 1, "sm_fraction": "7/7"},
}

SLICE_USABLE_HEADROOM = 0.90  # 10% headroom for CUDA context / runtime buffers
DEVICE_USABLE_HEADROOM = 0.92


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
        return asdict(self)


def is_mig_supported(gpu_sku: str) -> bool:
    """
    Evaluates whether the given GPU hardware SKU supports NVIDIA Multi-Instance GPU (MIG).
    H100, A100, A30, H200, B100, B200 support MIG.
    L40S, A10G, T4, L4, V100, RTX do NOT.
    """
    if not isinstance(gpu_sku, str) or not gpu_sku.strip():
        return False
    sku_upper = gpu_sku.upper().strip()

    for mig in MIG_CAPABLE_GPU_KEYWORDS:
        if re.search(rf"(?<![A-Z0-9]){mig}(?![A-Z0-9])", sku_upper) or sku_upper.startswith(mig):
            return True

    if sku_upper.startswith("MIG-") or re.search(r"\b\d+G\.\d+GB\b", sku_upper):
        return True

    return False


def _resolve_device_vram_gb(gpu_sku: str) -> float:
    """Determines physical (or MIG slice) VRAM capacity in GiB/GB for a GPU SKU."""
    sku_upper = gpu_sku.upper().strip()

    # Explicit MIG slice (e.g., MIG-2g.20gb or 3g.40gb)
    mig_match = re.search(r"\b\d+G\.(\d+)GB\b", sku_upper)
    if mig_match:
        return float(mig_match.group(1))

    # Explicit <N>GB suffix (e.g., A100-40GB, H100-80GB, L40S-48GB, A10G-24GB)
    gb_match = re.search(r"[-_](\d+)GB\b", sku_upper)
    if gb_match:
        return float(gb_match.group(1))

    # Known GPU families without explicit memory suffix
    if "B100" in sku_upper or "B200" in sku_upper:
        return 180.0
    if "H200" in sku_upper:
        return 141.0
    if "H100" in sku_upper and "NVL" in sku_upper:
        return 94.0
    if "H100" in sku_upper or "A100" in sku_upper:
        return 80.0
    if "L40S" in sku_upper or "L40" in sku_upper or "RTX-6000" in sku_upper or "A6000" in sku_upper:
        return 48.0
    if "V100" in sku_upper:
        return 32.0 if "32" in sku_upper else 16.0
    if "A30" in sku_upper or "A10G" in sku_upper or "A10" in sku_upper or "L4" in sku_upper or "3090" in sku_upper or "4090" in sku_upper:
        return 24.0
    if "T4" in sku_upper:
        return 16.0

    raise ValueError(
        f"Cannot determine VRAM capacity for unrecognized GPU SKU '{gpu_sku}'. "
        "Include capacity (e.g., 'A100-80GB', 'A10G-24GB') or use a supported NVIDIA SKU."
    )


def _get_mig_profiles_for_sku(gpu_sku: str) -> Dict[str, Dict[str, Any]]:
    """Returns the hardware-supported MIG profile map for a MIG-capable GPU SKU."""
    sku_upper = gpu_sku.upper().strip()
    vram_gb = _resolve_device_vram_gb(gpu_sku)

    if "B100" in sku_upper or "B200" in sku_upper:
        return MIG_B200_180GB_PROFILES
    if "H200" in sku_upper:
        return MIG_H200_141GB_PROFILES
    if "A30" in sku_upper:
        return MIG_A30_24GB_PROFILES
    if "H100" in sku_upper and ("NVL" in sku_upper or vram_gb >= 90.0):
        return MIG_94GB_PROFILES
    if "A100" in sku_upper and vram_gb <= 40.0:
        return MIG_40GB_PROFILES
    return MIG_80GB_PROFILES


def _select_mig_profile(
    gpu_sku: str,
    model_vram_gb: float,
    require_multi_slice: bool = True,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Selects the smallest hardware-valid MIG profile on gpu_sku whose usable VRAM
    accommodates model_vram_gb. When require_multi_slice is True, excludes 1-slice
    full-GPU profiles (where Full-GPU without MIG overhead is preferable).
    """
    profiles = _get_mig_profiles_for_sku(gpu_sku)
    for profile, info in sorted(profiles.items(), key=lambda x: (x[1]["vram_gb"], -x[1]["max_slices"])):
        if require_multi_slice and info["max_slices"] < 2:
            continue
        if model_vram_gb <= info["vram_gb"] * SLICE_USABLE_HEADROOM:
            return profile, info
    return None, None


def _generate_mig_configmap(mig_profile: str, slices: int) -> str:
    """Generates mig-parted ConfigMap for the NVIDIA GPU Operator."""
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


def _evaluate_mps_capacity(device_vram_gb: float, model_vram_gb: float) -> Tuple[int, float]:
    """
    Evaluates whether NVIDIA k8s-device-plugin CUDA MPS sharing can host >= 2 replicas.
    NVIDIA's MPS plugin divides physical device memory equally across configured replicas
    via CUDA_MPS_PINNED_DEVICE_MEM_LIMIT (each client receives device_vram_gb / replicas).
    """
    per_client_at_2 = (device_vram_gb / 2.0) * SLICE_USABLE_HEADROOM
    if model_vram_gb > per_client_at_2:
        return 1, per_client_at_2
    return 2, per_client_at_2


def recommend_partitioning(
    gpu_sku: str = "H100-80GB",
    workload_type: str = "vllm",
    tenancy: str = "multi",
    model_vram_gb: float = 16.0,
) -> Dict[str, Any]:
    """
    Recommends GPU sharing mechanism (MIG, MPS, Time-Slicing, or Full-GPU)
    enforcing physical device capacity, per-family MIG profiles, and equal MPS memory division.
    """
    if not isinstance(gpu_sku, str) or not gpu_sku.strip():
        raise ValueError("gpu_sku must be a non-empty string.")
    if isinstance(model_vram_gb, bool):
        raise ValueError(f"model_vram_gb must be a positive finite number, got {model_vram_gb!r}")

    try:
        vram_req = float(model_vram_gb)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model_vram_gb must be numeric, got {model_vram_gb!r}") from exc

    if not math.isfinite(vram_req) or vram_req <= 0.0:
        raise ValueError(f"model_vram_gb must be a positive finite number (> 0), got {model_vram_gb!r}")

    norm_workload = (workload_type or "").strip().lower()
    if not norm_workload:
        raise ValueError("workload_type must be a non-empty string.")

    norm_tenancy = (tenancy or "").strip().lower()
    if norm_tenancy not in ("single", "multi"):
        raise ValueError(f"tenancy must be 'single' or 'multi', got {tenancy!r}")

    device_vram_gb = _resolve_device_vram_gb(gpu_sku)
    usable_device_vram_gb = device_vram_gb * DEVICE_USABLE_HEADROOM
    supports_mig = is_mig_supported(gpu_sku)
    is_already_mig_slice = gpu_sku.upper().strip().startswith("MIG-") or bool(
        re.search(r"\b\d+G\.\d+GB\b", gpu_sku.upper())
    )

    warnings: List[str] = []
    recommended_strategy: str = "Full-GPU"
    mig_profile: Optional[str] = None
    risk_level: str = "LOW"
    rationale: str = ""
    configmap_yaml: str = ""

    if norm_workload == "vllm":
        warnings.append(
            "CRITICAL: Do NOT use Time-Slicing for vLLM! vLLM pre-allocates visible physical VRAM "
            "(--gpu-memory-utilization) for weights and KV-cache. Under Time-Slicing, co-located pods "
            "view the entire physical VRAM and crash with torch.cuda.OutOfMemoryError."
        )

    # 0. Workload exceeds physical single-GPU capacity
    if vram_req > usable_device_vram_gb:
        min_gpus = max(2, math.ceil(vram_req / usable_device_vram_gb))
        recommended_strategy = "Full-GPU"
        mig_profile = None
        risk_level = "CRITICAL"
        rationale = (
            f"Workload requires {vram_req:.1f} GiB VRAM, exceeding single-device usable capacity "
            f"({usable_device_vram_gb:.1f} GiB out of {device_vram_gb:.0f} GB on '{gpu_sku}'). "
            f"GPU sharing cannot fit this workload; use multi-GPU Tensor Parallelism (>= {min_gpus}x Full-GPU) "
            "or a higher-memory SKU."
        )
        warnings.append(
            f"Capacity Exceeded: {vram_req:.1f} GiB exceeds '{gpu_sku}' usable VRAM ({usable_device_vram_gb:.1f} GiB). "
            f"Single-GPU deployment will OOM; shard across >= {min_gpus} GPUs."
        )
        configmap_yaml = _generate_full_gpu_configmap()

    # 0b. Already a pre-sliced MIG resource
    elif is_already_mig_slice:
        slice_match = re.search(r"(\d+g\.\d+gb)", gpu_sku.lower())
        recommended_strategy = "MIG"
        mig_profile = slice_match.group(1) if slice_match else None
        risk_level = "LOW"
        rationale = (
            f"Target '{gpu_sku}' is already a hardware-isolated MIG slice ({device_vram_gb:.0f} GB) and fits "
            f"the {vram_req:.1f} GiB workload within its {usable_device_vram_gb:.1f} GiB usable budget."
        )
        warnings.append(
            "Existing MIG slices cannot be further sub-partitioned via MPS or nested MIG."
        )
        configmap_yaml = _generate_mig_configmap(mig_profile or "1g.10gb", slices=1)

    # 1. Dev Notebooks / Interactive Sandboxes
    elif norm_workload == "dev-notebook":
        if vram_req <= device_vram_gb * 0.22:
            recommended_strategy = "Time-Slicing"
            risk_level = "LOW"
            rationale = (
                "Interactive development notebooks have bursty, idle-heavy access patterns and a small "
                f"per-session footprint ({vram_req:.1f} GiB vs {device_vram_gb:.0f} GB device). "
                "Time-Slicing (replicas: 4) maximizes developer density."
            )
            warnings.append(
                "Time-Slicing does not enforce memory isolation between containers; concurrent memory spikes "
                f"exceeding {device_vram_gb:.0f} GB total will cause CUDA OOM."
            )
            configmap_yaml = _generate_time_slicing_configmap(replicas=4)
        elif vram_req <= device_vram_gb * 0.45:
            recommended_strategy = "Time-Slicing"
            risk_level = "MEDIUM"
            rationale = (
                f"Notebook working set ({vram_req:.1f} GiB) consumes nearly half of '{gpu_sku}' ({device_vram_gb:.0f} GB). "
                "Time-Slicing is capped at replicas=2 to reduce mutual OOM collisions."
            )
            warnings.append(
                "Time-Slicing does not isolate VRAM; cap replicas to 2 or use MIG if deterministic isolation is needed."
            )
            configmap_yaml = _generate_time_slicing_configmap(replicas=2)
        else:
            recommended_strategy = "Full-GPU"
            risk_level = "MEDIUM" if norm_tenancy == "multi" else "LOW"
            rationale = (
                f"Notebook requires {vram_req:.1f} GiB on '{gpu_sku}' ({device_vram_gb:.0f} GB), leaving insufficient "
                "headroom for even 2 concurrent Time-Slicing tenants. Dedicated Full-GPU allocation is required."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 2. vLLM Serving Workloads
    elif norm_workload == "vllm":
        if norm_tenancy == "multi":
            if supports_mig:
                mig_profile, prof_info = _select_mig_profile(gpu_sku, vram_req, require_multi_slice=True)
                if mig_profile and prof_info:
                    recommended_strategy = "MIG"
                    risk_level = "LOW"
                    rationale = (
                        f"Hardware MIG profile '{mig_profile}' ({prof_info['vram_gb']:.0f} GB, up to "
                        f"{prof_info['max_slices']} slices/GPU) provides isolated SMs, memory controllers, and L2 cache "
                        f"for {vram_req:.1f} GiB vLLM instances without co-tenant OOM collisions."
                    )
                    configmap_yaml = _generate_mig_configmap(mig_profile, slices=prof_info["max_slices"])
                else:
                    recommended_strategy = "Full-GPU"
                    mig_profile = None
                    risk_level = "LOW"
                    rationale = (
                        f"Workload ({vram_req:.1f} GiB) exceeds multi-slice MIG partitions on '{gpu_sku}'. "
                        "Dedicated Full-GPU allocation per replica provides full HBM capacity and bandwidth."
                    )
                    configmap_yaml = _generate_full_gpu_configmap()
            else:
                recommended_strategy = "Full-GPU"
                mig_profile = None
                risk_level = "HIGH"
                rationale = (
                    f"GPU SKU '{gpu_sku}' does not support hardware MIG partitioning. Because vLLM eagerly "
                    "pre-allocates VRAM, Dedicated Full-GPU allocation per replica is required for stability."
                )
                warnings.append(
                    f"GPU '{gpu_sku}' lacks silicon MIG support; dedicate 1 physical GPU per vLLM replica."
                )
                configmap_yaml = _generate_full_gpu_configmap()
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = (
                "Dedicated single-tenant vLLM serving. Full-GPU allocation maximizes KV-cache capacity and bandwidth."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 3. Embeddings, RAG Pipelines, and Triton Inference Server
    elif norm_workload in ("embeddings", "rag-pipeline", "triton"):
        if norm_tenancy == "multi":
            if supports_mig:
                mig_profile, prof_info = _select_mig_profile(gpu_sku, vram_req, require_multi_slice=True)
                if mig_profile and prof_info:
                    recommended_strategy = "MIG"
                    risk_level = "LOW"
                    rationale = (
                        f"Hardware MIG partition '{mig_profile}' ({prof_info['vram_gb']:.0f} GB, "
                        f"{prof_info['max_slices']}x per GPU) isolates {norm_workload} instances with dedicated "
                        "SMs and HBM bandwidth."
                    )
                    configmap_yaml = _generate_mig_configmap(mig_profile, slices=prof_info["max_slices"])
                else:
                    recommended_strategy = "Full-GPU"
                    mig_profile = None
                    risk_level = "LOW"
                    rationale = (
                        f"Workload requires {vram_req:.1f} GiB on '{gpu_sku}' ({device_vram_gb:.0f} GB), exceeding "
                        "multi-tenant MIG slice sizes. Dedicated Full-GPU allocation is required."
                    )
                    configmap_yaml = _generate_full_gpu_configmap()
            else:
                mps_replicas, per_client_limit = _evaluate_mps_capacity(device_vram_gb, vram_req)
                if mps_replicas >= 2:
                    recommended_strategy = "MPS"
                    mig_profile = None
                    risk_level = "MEDIUM" if norm_workload == "triton" else "LOW"
                    rationale = (
                        f"GPU '{gpu_sku}' lacks MIG, and each {vram_req:.1f} GiB replica fits comfortably within "
                        f"NVIDIA MPS equal per-client memory partitioning (~{per_client_limit:.1f} GiB usable per "
                        f"replica at replicas={mps_replicas} on {device_vram_gb:.0f} GB)."
                    )
                    warnings.append(
                        "CUDA MPS shares a unified fault domain: an uncontained GPU memory fault in one client "
                        "terminates co-located MPS processes."
                    )
                    configmap_yaml = _generate_mps_configmap(replicas=mps_replicas)
                else:
                    recommended_strategy = "Full-GPU"
                    mig_profile = None
                    risk_level = "HIGH"
                    rationale = (
                        f"NVIDIA's Kubernetes MPS device plugin divides '{gpu_sku}' ({device_vram_gb:.0f} GB) equally "
                        f"across replicas (~{device_vram_gb / 2.0:.1f} GB raw / ~{per_client_limit:.1f} GiB usable "
                        f"at replicas=2), which cannot fit a {vram_req:.1f} GiB {norm_workload} workload. "
                        "Dedicated Full-GPU allocation is required."
                    )
                    warnings.append(
                        f"MPS Sharing Infeasible: 2 replicas on {device_vram_gb:.0f} GB cap each client at "
                        f"~{device_vram_gb / 2.0:.1f} GB (< {vram_req:.1f} GiB required)."
                    )
                    configmap_yaml = _generate_full_gpu_configmap()
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = (
                f"Single-tenant dedicated deployment for {norm_workload} ({vram_req:.1f} GiB) on '{gpu_sku}'."
            )
            configmap_yaml = _generate_full_gpu_configmap()

    # 4. Generic / Other Workloads
    else:
        if supports_mig and norm_tenancy == "multi":
            mig_profile, prof_info = _select_mig_profile(gpu_sku, vram_req, require_multi_slice=True)
            if mig_profile and prof_info:
                recommended_strategy = "MIG"
                risk_level = "LOW"
                rationale = (
                    f"MIG partitioning ('{mig_profile}', {prof_info['vram_gb']:.0f} GB) provides hardware isolation "
                    f"for multi-tenant {norm_workload} workloads."
                )
                configmap_yaml = _generate_mig_configmap(mig_profile, slices=prof_info["max_slices"])
            else:
                recommended_strategy = "Full-GPU"
                mig_profile = None
                risk_level = "LOW"
                rationale = (
                    f"Workload ({vram_req:.1f} GiB) exceeds multi-slice MIG partitions on '{gpu_sku}'; "
                    "Full-GPU allocation is required."
                )
                configmap_yaml = _generate_full_gpu_configmap()
        else:
            recommended_strategy = "Full-GPU"
            mig_profile = None
            risk_level = "LOW"
            rationale = "Dedicated Full-GPU allocation ensures uncontended GPU memory and compute access."
            configmap_yaml = _generate_full_gpu_configmap()

    rec = PartitionRecommendation(
        gpu_sku=gpu_sku,
        supports_mig=supports_mig,
        workload_type=norm_workload,
        tenancy=norm_tenancy,
        model_vram_gb=vram_req,
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

    res = recommend_partitioning(
        gpu_sku="H100-80GB",
        workload_type="vllm",
        tenancy="multi",
        model_vram_gb=16.0,
    )
    print(json.dumps(res, indent=2))
