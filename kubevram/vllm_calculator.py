"""
KubeVRAM vLLM Resource & KV-Cache Sizing Calculator.

Static VRAM & KV-cache capacity model for:
1. Model weight footprint based on precision (FP32, BF16, FP16, FP8, INT8, INT4/AWQ/GPTQ) with decimal-to-GiB conversion.
2. Per-token KV cache memory consumption (GQA-aware, accounting for TP head replication).
3. Usable VRAM after CUDA context and CUDA Graph capture overhead.
4. Maximum sequence concurrency at target context length.
5. Minimum fitting Tensor Parallelism (TP) size and validated vLLM 0.6.4 CLI flags.
6. Hardware-targeted Kubernetes Deployment (single-node) or LeaderWorkerSet (multinode) manifest.

Note: Sizing establishes static VRAM feasibility and KV-cache token budgets; it does not model
request-length distributions, interconnect latency, or empirical throughput optimality.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Built-in catalog of production LLM architectures
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "params_b": 8.03,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "llama-3.1-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "params_b": 70.6,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "llama-3.1-405b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "params_b": 405.0,
        "num_layers": 126,
        "num_attention_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.24,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 32768,
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.5,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "deepseek-r1-32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "params_b": 32.5,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "mixtral-8x22b": {
        "hf_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "params_b": 141.0,
        "num_layers": 56,
        "num_attention_heads": 48,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 65536,
    },
}

# Physical and MIG GPU VRAM catalog (in MiB)
GPU_CATALOG: Dict[str, int] = {
    "H100-80GB": 81920,
    "A100-80GB": 81920,
    "A100-40GB": 40960,
    "L40S-48GB": 49152,
    "A10G-24GB": 24576,
    "MIG-3g.40gb": 40960,
    "MIG-2g.20gb": 20480,
    "MIG-1g.10gb": 10240,
}

# Hardware SKU metadata for nodeAffinity targeting, FP8 support, and topology limits
GPU_SKU_METADATA: Dict[str, Dict[str, Any]] = {
    "H100-80GB": {
        "compute_capability": (9, 0),
        "supports_fp8_w8a8": True,
        "interconnect": "nvswitch",
        "max_gpus_per_node": 8,
        "gfd_product_labels": [
            "NVIDIA-H100-80GB-HBM3",
            "NVIDIA-H100-SXM5-80GB",
            "NVIDIA-H100-PCIe-80GB",
        ],
    },
    "A100-80GB": {
        "compute_capability": (8, 0),
        "supports_fp8_w8a8": False,
        "interconnect": "nvlink",
        "max_gpus_per_node": 8,
        "gfd_product_labels": [
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-A100-80GB-PCIe",
        ],
    },
    "A100-40GB": {
        "compute_capability": (8, 0),
        "supports_fp8_w8a8": False,
        "interconnect": "nvlink",
        "max_gpus_per_node": 8,
        "gfd_product_labels": [
            "NVIDIA-A100-SXM4-40GB",
            "NVIDIA-A100-PCIE-40GB",
        ],
    },
    "L40S-48GB": {
        "compute_capability": (8, 9),
        "supports_fp8_w8a8": True,
        "interconnect": "pcie",
        "max_gpus_per_node": 8,
        "gfd_product_labels": [
            "NVIDIA-L40S",
            "NVIDIA-L40S-48GB",
        ],
    },
    "A10G-24GB": {
        "compute_capability": (8, 6),
        "supports_fp8_w8a8": False,
        "interconnect": "pcie",
        "max_gpus_per_node": 8,
        "gfd_product_labels": [
            "NVIDIA-A10G",
        ],
    },
    "MIG-3g.40gb": {
        "compute_capability": (8, 0),
        "supports_fp8_w8a8": False,
        "interconnect": "mig",
        "max_gpus_per_node": 1,
        "gfd_product_labels": [
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-A100-80GB-PCIe",
            "NVIDIA-H100-80GB-HBM3",
        ],
    },
    "MIG-2g.20gb": {
        "compute_capability": (8, 0),
        "supports_fp8_w8a8": False,
        "interconnect": "mig",
        "max_gpus_per_node": 1,
        "gfd_product_labels": [
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-A100-80GB-PCIe",
            "NVIDIA-H100-80GB-HBM3",
        ],
    },
    "MIG-1g.10gb": {
        "compute_capability": (8, 0),
        "supports_fp8_w8a8": False,
        "interconnect": "mig",
        "max_gpus_per_node": 1,
        "gfd_product_labels": [
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-A100-80GB-PCIe",
            "NVIDIA-H100-80GB-HBM3",
        ],
    },
}

# Weight precision specification: (bytes_per_param, scale_overhead_factor, vllm_dtype_flag, vllm_quantization_flag)
DTYPE_SPEC_MAP: Dict[str, Tuple[float, float, str, Optional[str]]] = {
    "auto": (2.0, 1.0, "auto", None),
    "fp32": (4.0, 1.0, "float32", None),
    "float32": (4.0, 1.0, "float32", None),
    "float": (4.0, 1.0, "float32", None),
    "fp16": (2.0, 1.0, "float16", None),
    "float16": (2.0, 1.0, "float16", None),
    "half": (2.0, 1.0, "float16", None),
    "bf16": (2.0, 1.0, "bfloat16", None),
    "bfloat16": (2.0, 1.0, "bfloat16", None),
    "fp8": (1.0, 1.04, "auto", "fp8"),
    "fp8_e4m3": (1.0, 1.04, "auto", "fp8"),
    "fp8_e5m2": (1.0, 1.04, "auto", "fp8"),
    "int8": (1.0, 1.04, "auto", "compressed-tensors"),
    "awq": (0.5, 1.08, "float16", "awq"),
    "gptq": (0.5, 1.08, "float16", "gptq"),
    "int4": (0.5, 1.08, "bfloat16", "bitsandbytes"),
    "nf4": (0.5, 1.08, "bfloat16", "bitsandbytes"),
    "bitsandbytes": (0.5, 1.08, "bfloat16", "bitsandbytes"),
}

# Backward-compatible bytes-per-param lookup
DTYPE_BYTES_MAP: Dict[str, float] = {k: v[0] for k, v in DTYPE_SPEC_MAP.items()}

VALID_VLLM_DTYPES = {"auto", "half", "float16", "bfloat16", "float", "float32"}
VALID_VLLM_KV_DTYPES = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}

CUDA_CONTEXT_GRAPH_OVERHEAD_GIB: float = 2.8
BYTES_PER_GIB: float = float(1024**3)
PARAMS_PER_BILLION: float = 1_000_000_000.0


@dataclass
class VLLMTuningResult:
    """Calculated tuning parameters and Kubernetes deployment specs for a vLLM workload."""

    model_key: str
    hf_id: str
    gpu_sku: str
    num_gpus: int
    dtype: str
    kv_dtype: str
    target_context: int
    target_concurrency: int
    gpu_memory_utilization: float
    gpu_capacity_gib: float
    total_gpu_capacity_gib: float
    weight_memory_gib: float
    weight_memory_per_gpu_gib: float
    cuda_overhead_gib: float
    total_cuda_overhead_gib: float
    usable_vram_gib: float
    total_usable_vram_gib: float
    available_kv_cache_gib: float
    per_token_kv_bytes: int
    max_total_kv_tokens: int
    max_concurrency_at_target_context: int
    recommended_max_model_len: int
    recommended_tp_size: Optional[int]
    fits_in_vram: bool
    warnings: List[str] = field(default_factory=list)
    vllm_cli_args: List[str] = field(default_factory=list)
    k8s_manifest_yaml: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary matching KubeVRAM standard schema."""
        data = asdict(self)
        data["weight_memory_gb"] = self.weight_memory_gib
        data["kv_cache_gib"] = self.available_kv_cache_gib
        data["cuda_context_overhead_gib"] = self.cuda_overhead_gib
        return data


def _resolve_model(model_key: str) -> Tuple[str, Dict[str, Any]]:
    """Resolves a model key or case-insensitive alias to catalog entry."""
    norm_key = model_key.strip().lower()
    if norm_key in MODEL_CATALOG:
        return norm_key, MODEL_CATALOG[norm_key]

    aliases = {
        "llama-3-8b": "llama-3.1-8b",
        "llama3-8b": "llama-3.1-8b",
        "llama3.1-8b": "llama-3.1-8b",
        "llama-3-70b": "llama-3.1-70b",
        "llama3-70b": "llama-3.1-70b",
        "llama3.1-70b": "llama-3.1-70b",
        "llama-3-405b": "llama-3.1-405b",
        "llama3-405b": "llama-3.1-405b",
        "llama3.1-405b": "llama-3.1-405b",
        "mistral": "mistral-7b",
        "qwen-32b": "qwen2.5-32b",
        "qwen2.5": "qwen2.5-32b",
        "deepseek-32b": "deepseek-r1-32b",
        "deepseek-r1": "deepseek-r1-32b",
        "mixtral": "mixtral-8x22b",
    }
    if norm_key in aliases:
        canonical = aliases[norm_key]
        return canonical, MODEL_CATALOG[canonical]

    raise KeyError(
        f"Unknown model key '{model_key}'. Available models: {list(MODEL_CATALOG.keys())}"
    )


def _resolve_gpu_sku(gpu_sku: str) -> Tuple[str, int]:
    """Resolves GPU SKU string to canonical name and MiB capacity."""
    norm_sku = gpu_sku.strip()
    if norm_sku in GPU_CATALOG:
        return norm_sku, GPU_CATALOG[norm_sku]

    for k, v in GPU_CATALOG.items():
        if k.lower() == norm_sku.lower():
            return k, v

    aliases = {
        "h100": "H100-80GB",
        "a100": "A100-80GB",
        "a100-80": "A100-80GB",
        "a100-40": "A100-40GB",
        "l40s": "L40S-48GB",
        "a10g": "A10G-24GB",
        "3g.40gb": "MIG-3g.40gb",
        "2g.20gb": "MIG-2g.20gb",
        "1g.10gb": "MIG-1g.10gb",
    }
    if norm_sku.lower() in aliases:
        canonical = aliases[norm_sku.lower()]
        return canonical, GPU_CATALOG[canonical]

    raise KeyError(
        f"Unknown GPU SKU '{gpu_sku}'. Available SKUs: {list(GPU_CATALOG.keys())}"
    )


def _resolve_dtypes(
    dtype: str, kv_dtype: str
) -> Tuple[str, float, float, str, Optional[str], str, float, str]:
    """
    Validates and resolves weight dtype and KV cache dtype into byte widths and
    executable vLLM 0.6.4 CLI flag values. Raises ValueError on unsupported dtypes.
    """
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("dtype must be a non-empty string.")
    if not isinstance(kv_dtype, str) or not kv_dtype.strip():
        raise ValueError("kv_dtype must be a non-empty string.")

    norm_dtype = dtype.strip().lower()
    if norm_dtype not in DTYPE_SPEC_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported dtypes: {sorted(DTYPE_SPEC_MAP.keys())}"
        )
    bytes_per_param, overhead_factor, vllm_dtype, vllm_quant = DTYPE_SPEC_MAP[norm_dtype]

    norm_kv_dtype = kv_dtype.strip().lower()
    is_fp32_compute = vllm_dtype == "float32"

    if norm_kv_dtype == "auto":
        kv_bytes = 4.0 if is_fp32_compute else 2.0
        vllm_kv_dtype = "auto"
    elif norm_kv_dtype in ("fp16", "float16", "half", "bf16", "bfloat16"):
        if is_fp32_compute:
            raise ValueError(
                f"kv_dtype '{kv_dtype}' is incompatible with 32-bit compute dtype '{dtype}' in vLLM 0.6.4 "
                "(supported --kv-cache-dtype choices: auto, fp8, fp8_e4m3, fp8_e5m2)."
            )
        kv_bytes = 2.0
        vllm_kv_dtype = "auto"
    elif norm_kv_dtype in ("fp32", "float32", "float"):
        if not is_fp32_compute:
            raise ValueError(
                f"kv_dtype '{kv_dtype}' requires 32-bit weight/compute dtype in vLLM 0.6.4."
            )
        kv_bytes = 4.0
        vllm_kv_dtype = "auto"
    elif norm_kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        kv_bytes = 1.0
        vllm_kv_dtype = norm_kv_dtype
    else:
        raise ValueError(
            f"Unsupported kv_dtype '{kv_dtype}'. Supported kv_dtypes: "
            "['auto', 'bfloat16', 'bf16', 'float16', 'fp16', 'float32', 'fp32', 'fp8', 'fp8_e4m3', 'fp8_e5m2']"
        )

    return (
        norm_dtype,
        bytes_per_param,
        overhead_factor,
        vllm_dtype,
        vllm_quant,
        norm_kv_dtype,
        kv_bytes,
        vllm_kv_dtype,
    )


def _get_gpu_memory_utilization(gpu_capacity_gib: float) -> float:
    """
    Returns recommended vLLM --gpu-memory-utilization factor.
    Leaves headroom for PyTorch NCCL buffers, CUDA context, and CUDA graphs.
    """
    if gpu_capacity_gib >= 40.0:
        return 0.92
    elif gpu_capacity_gib <= 24.0:
        return 0.88
    return 0.90


def _is_tp_valid_for_model(
    model_info: Dict[str, Any],
    tp: int,
    gpu_sku: str,
    vllm_quant: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Checks vLLM 0.6.4 Tensor Parallelism compatibility invariants."""
    if tp < 1:
        return False, f"Invalid tensor parallel size {tp}; must be >= 1."
    if gpu_sku.startswith("MIG-") and tp > 1:
        return (
            False,
            f"CUDA/NVML forbids multi-device Tensor Parallelism across {tp} MIG slices "
            "(CUDA_VISIBLE_DEVICES supports at most 1 MIG instance per process).",
        )
    if vllm_quant == "bitsandbytes" and tp > 1:
        return (
            False,
            f"vLLM 0.6.4 bitsandbytes quantization does not support tensor-parallel-size={tp} (TP must be 1).",
        )
    num_attn = model_info["num_attention_heads"]
    if num_attn % tp != 0:
        return (
            False,
            f"Model num_attention_heads ({num_attn}) is not divisible by tensor-parallel-size ({tp}).",
        )
    num_kv = model_info["num_kv_heads"]
    if num_kv % tp != 0 and tp % num_kv != 0:
        return (
            False,
            f"Model num_kv_heads ({num_kv}) must be divisible by tensor-parallel-size ({tp}), "
            f"or tensor-parallel-size ({tp}) must be a multiple of num_kv_heads ({num_kv}).",
        )
    return True, None


def _per_gpu_kv_token_bytes(
    model_info: Dict[str, Any], tp: int, kv_bytes: float
) -> Tuple[int, int]:
    """
    Computes KV heads per worker and per-GPU per-token KV cache bytes matching vLLM's
    ModelConfig.get_num_kv_heads: each worker holds at least 1 KV head when TP > num_kv_heads.
    """
    kv_heads_per_gpu = max(1, math.ceil(model_info["num_kv_heads"] / tp))
    per_gpu_bytes = int(
        2 * model_info["num_layers"] * kv_heads_per_gpu * model_info["head_dim"] * kv_bytes
    )
    return kv_heads_per_gpu, per_gpu_bytes


def _calculate_optimal_tp(
    model_info: Dict[str, Any],
    gpu_sku: str,
    vllm_quant: Optional[str],
    weight_memory_gib: float,
    gpu_capacity_gib: float,
    kv_bytes: float,
    target_context: int,
    target_concurrency: int,
) -> Optional[int]:
    """
    Calculates minimum valid Tensor Parallelism (TP) size from {1, 2, 4, 8, 16}
    that satisfies target_context (and preferably target_concurrency).
    Returns None if no candidate TP up to 16 fits in VRAM.
    """
    candidate_tps = [1] if gpu_sku.startswith("MIG-") else [1, 2, 4, 8, 16]
    utilization = _get_gpu_memory_utilization(gpu_capacity_gib)
    usable_per_gpu = gpu_capacity_gib * utilization
    target_tokens = target_context * max(1, target_concurrency)

    min_fitting_tp: Optional[int] = None

    for tp in candidate_tps:
        valid_tp, _ = _is_tp_valid_for_model(model_info, tp, gpu_sku, vllm_quant)
        if not valid_tp:
            continue
        weight_per_gpu = weight_memory_gib / tp
        kv_cache_per_gpu = usable_per_gpu - weight_per_gpu - CUDA_CONTEXT_GRAPH_OVERHEAD_GIB
        if kv_cache_per_gpu >= 0.5:
            _, per_gpu_kv_bytes = _per_gpu_kv_token_bytes(model_info, tp, kv_bytes)
            total_tokens = int((kv_cache_per_gpu * BYTES_PER_GIB) // per_gpu_kv_bytes)
            if total_tokens >= target_context:
                if min_fitting_tp is None:
                    min_fitting_tp = tp
                if total_tokens >= target_tokens:
                    return tp

    return min_fitting_tp


def _build_node_affinity_yaml(gpu_sku: str, indent: int = 6) -> str:
    """Builds Kubernetes nodeAffinity YAML enforcing exact GPU SKU and sharing-strategy constraints."""
    pad = " " * indent
    meta = GPU_SKU_METADATA.get(gpu_sku, {})
    product_labels = meta.get("gfd_product_labels", [])
    is_mig = gpu_sku.upper().startswith("MIG-")

    lines = [
        f"{pad}affinity:",
        f"{pad}  nodeAffinity:",
        f"{pad}    requiredDuringSchedulingIgnoredDuringExecution:",
        f"{pad}      nodeSelectorTerms:",
        f"{pad}        - matchExpressions:",
    ]
    if product_labels:
        lines.append(f"{pad}            - key: nvidia.com/gpu.product")
        lines.append(f"{pad}              operator: In")
        lines.append(f"{pad}              values:")
        for lbl in product_labels:
            lines.append(f"{pad}                - \"{lbl}\"")
    if not is_mig:
        lines.append(f"{pad}            - key: nvidia.com/gpu.sharing-strategy")
        lines.append(f"{pad}              operator: NotIn")
        lines.append(f"{pad}              values:")
        lines.append(f"{pad}                - \"time-slicing\"")
        lines.append(f"{pad}                - \"mps\"")
    return "\n".join(lines)


def _generate_k8s_deployment_manifest(
    deployment_name: str,
    hf_id: str,
    gpu_sku: str,
    num_gpus: int,
    vllm_cli_args: List[str],
) -> str:
    """
    Generates Kubernetes manifest for vLLM serving:
    - Single-node Deployment for num_gpus <= 8 (with SKU nodeAffinity, /dev/shm, RBAC SA, Gang Scheduling).
    - Multi-node LeaderWorkerSet (Ray backend) when num_gpus > 8 so no pod requests > 8 GPUs.
    """
    is_mig = gpu_sku.upper().startswith("MIG-")
    gpu_resource_name = f"nvidia.com/{gpu_sku.lower()}" if is_mig else "nvidia.com/gpu"
    sa_name = f"{deployment_name}-sa"

    sa_manifest = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: {sa_name}
  namespace: ai-inference
automountServiceAccountToken: false
---
"""

    if num_gpus <= 8:
        cpu_req = max(8, 4 * num_gpus)
        cpu_lim = max(16, 8 * num_gpus)
        mem_req_gi = max(32, 16 * num_gpus)
        mem_lim_gi = max(64, 32 * num_gpus)

        affinity_yaml = _build_node_affinity_yaml(gpu_sku, indent=6)
        args_formatted = "\n".join(f"            - \"{arg}\"" for arg in vllm_cli_args)

        gang_meta = ""
        if num_gpus > 1:
            gang_meta = (
                "      annotations:\n"
                f"        scheduling.k8s.io/group-name: {deployment_name}-gang\n"
            )
            pod_labels = (
                f"        app: {deployment_name}\n"
                f"        pod-group: {deployment_name}-gang"
            )
        else:
            pod_labels = f"        app: {deployment_name}"

        return sa_manifest + f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: ai-inference
  labels:
    app.kubernetes.io/name: {deployment_name}
    app.kubernetes.io/part-of: kubevram
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
{gang_meta}      labels:
{pod_labels}
    spec:
      serviceAccountName: {sa_name}
      automountServiceAccountToken: false
      runtimeClassName: nvidia
{affinity_yaml}
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.4
          args:
{args_formatted}
          securityContext:
            privileged: false
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 45
            periodSeconds: 10
          resources:
            requests:
              cpu: "{cpu_req}"
              memory: "{mem_req_gi}Gi"
              {gpu_resource_name}: "{num_gpus}"
            limits:
              cpu: "{cpu_lim}"
              memory: "{mem_lim_gi}Gi"
              {gpu_resource_name}: "{num_gpus}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
"""

    # Multinode LeaderWorkerSet arrangement (e.g., TP=16 across 2x 8-GPU nodes)
    gpus_per_node = 8 if num_gpus % 8 == 0 else math.gcd(num_gpus, 8)
    num_nodes = num_gpus // gpus_per_node
    cpu_req = max(8, 4 * gpus_per_node)
    cpu_lim = max(16, 8 * gpus_per_node)
    mem_req_gi = max(32, 16 * gpus_per_node)
    mem_lim_gi = max(64, 32 * gpus_per_node)

    affinity_yaml = _build_node_affinity_yaml(gpu_sku, indent=10)
    ray_cli_args = list(vllm_cli_args)
    if not any(a.startswith("--distributed-executor-backend=") for a in ray_cli_args):
        ray_cli_args.append("--distributed-executor-backend=ray")
    joined_vllm_cmd = " ".join(ray_cli_args)

    return sa_manifest + f"""apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: {deployment_name}
  namespace: ai-inference
  labels:
    app.kubernetes.io/name: {deployment_name}
    app.kubernetes.io/part-of: kubevram
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: {num_nodes}
    leaderTemplate:
      metadata:
        annotations:
          scheduling.k8s.io/group-name: {deployment_name}-gang
        labels:
          app: {deployment_name}
          pod-group: {deployment_name}-gang
      spec:
        serviceAccountName: {sa_name}
        automountServiceAccountToken: false
        runtimeClassName: nvidia
{affinity_yaml}
        containers:
          - name: vllm-leader
            image: vllm/vllm-openai:v0.6.4
            command:
              - /bin/bash
              - -lc
              - "ray start --head --port=6379 && python3 -m vllm.entrypoints.openai.api_server {joined_vllm_cmd}"
            securityContext:
              privileged: false
              allowPrivilegeEscalation: false
              capabilities:
                drop:
                  - ALL
            ports:
              - containerPort: 8000
                name: http
                protocol: TCP
            resources:
              requests:
                cpu: "{cpu_req}"
                memory: "{mem_req_gi}Gi"
                {gpu_resource_name}: "{gpus_per_node}"
              limits:
                cpu: "{cpu_lim}"
                memory: "{mem_lim_gi}Gi"
                {gpu_resource_name}: "{gpus_per_node}"
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 16Gi
    workerTemplate:
      metadata:
        annotations:
          scheduling.k8s.io/group-name: {deployment_name}-gang
        labels:
          app: {deployment_name}
          pod-group: {deployment_name}-gang
      spec:
        serviceAccountName: {sa_name}
        automountServiceAccountToken: false
        runtimeClassName: nvidia
{affinity_yaml}
        containers:
          - name: vllm-worker
            image: vllm/vllm-openai:v0.6.4
            command:
              - /bin/bash
              - -lc
              - "ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"
            securityContext:
              privileged: false
              allowPrivilegeEscalation: false
              capabilities:
                drop:
                  - ALL
            resources:
              requests:
                cpu: "{cpu_req}"
                memory: "{mem_req_gi}Gi"
                {gpu_resource_name}: "{gpus_per_node}"
              limits:
                cpu: "{cpu_lim}"
                memory: "{mem_lim_gi}Gi"
                {gpu_resource_name}: "{gpus_per_node}"
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 16Gi
"""


def calculate_vllm_tuning(
    model_key: str = "llama-3.1-70b",
    gpu_sku: str = "H100-80GB",
    num_gpus: Optional[int] = None,
    dtype: str = "bfloat16",
    kv_dtype: str = "auto",
    target_context: int = 16384,
    target_concurrency: int = 32,
) -> Dict[str, Any]:
    """
    Computes static VRAM footprints, GQA/TP-aware KV cache token limits, concurrency capacity,
    validated vLLM 0.6.4 CLI flags, and hardware-targeted Kubernetes manifests.

    When num_gpus is None, selects the minimum fitting TP size automatically so sizing
    fields and emitted deployment artifacts always describe the exact same configuration.
    When an explicit num_gpus is infeasible, fits_in_vram is False and no executable CLI
    arguments or manifest are emitted.
    """
    canonical_model_key, model_info = _resolve_model(model_key)
    canonical_gpu_sku, gpu_mib = _resolve_gpu_sku(gpu_sku)
    sku_meta = GPU_SKU_METADATA[canonical_gpu_sku]

    if isinstance(target_context, bool) or int(target_context) < 16:
        raise ValueError(f"target_context must be an integer >= 16, got {target_context!r}")
    if isinstance(target_concurrency, bool) or int(target_concurrency) < 1:
        raise ValueError(f"target_concurrency must be an integer >= 1, got {target_concurrency!r}")
    target_context = int(target_context)
    target_concurrency = int(target_concurrency)

    if num_gpus is not None:
        if isinstance(num_gpus, bool) or int(num_gpus) < 1:
            raise ValueError(f"num_gpus must be an integer >= 1, got {num_gpus!r}")
        explicit_num_gpus: Optional[int] = int(num_gpus)
    else:
        explicit_num_gpus = None

    (
        norm_dtype,
        bytes_per_param,
        overhead_factor,
        vllm_dtype,
        vllm_quant,
        norm_kv_dtype,
        kv_bytes,
        vllm_kv_dtype,
    ) = _resolve_dtypes(dtype, kv_dtype)

    # 1. Exact Model Weight Footprint (decimal parameter count -> binary GiB)
    raw_weight_bytes = model_info["params_b"] * PARAMS_PER_BILLION * bytes_per_param * overhead_factor
    weight_memory_gib = round(raw_weight_bytes / BYTES_PER_GIB, 3)

    # 2. GPU Capacity & Minimum Fitting TP
    gpu_capacity_gib = round(gpu_mib / 1024.0, 3)
    gpu_memory_utilization = _get_gpu_memory_utilization(gpu_capacity_gib)
    usable_vram_gib = round(gpu_capacity_gib * gpu_memory_utilization, 3)

    recommended_tp_size = _calculate_optimal_tp(
        model_info=model_info,
        gpu_sku=canonical_gpu_sku,
        vllm_quant=vllm_quant,
        weight_memory_gib=weight_memory_gib,
        gpu_capacity_gib=gpu_capacity_gib,
        kv_bytes=kv_bytes,
        target_context=target_context,
        target_concurrency=target_concurrency,
    )

    if explicit_num_gpus is not None:
        resolved_num_gpus = explicit_num_gpus
    elif recommended_tp_size is not None:
        resolved_num_gpus = recommended_tp_size
    else:
        resolved_num_gpus = 1

    weight_memory_per_gpu_gib = round(weight_memory_gib / resolved_num_gpus, 3)
    cuda_overhead_gib = CUDA_CONTEXT_GRAPH_OVERHEAD_GIB
    total_cuda_overhead_gib = round(cuda_overhead_gib * resolved_num_gpus, 3)
    total_gpu_capacity_gib = round(gpu_capacity_gib * resolved_num_gpus, 3)
    total_usable_vram_gib = round(usable_vram_gib * resolved_num_gpus, 3)

    # 3. Available KV Cache Memory
    kv_cache_gib_per_gpu = usable_vram_gib - weight_memory_per_gpu_gib - cuda_overhead_gib
    if kv_cache_gib_per_gpu > 0:
        available_kv_cache_gib = round(kv_cache_gib_per_gpu * resolved_num_gpus, 3)
    else:
        available_kv_cache_gib = 0.0

    # 4. Per-Token KV Cache Bytes (GQA + TP KV-head replication aware)
    kv_heads_per_gpu, per_gpu_token_kv_bytes = _per_gpu_kv_token_bytes(
        model_info, resolved_num_gpus, kv_bytes
    )
    per_token_kv_bytes = per_gpu_token_kv_bytes * resolved_num_gpus

    # 5. Max KV Tokens Across TP Group (all TP ranks store their KV head shard for every token)
    if kv_cache_gib_per_gpu > 0 and per_gpu_token_kv_bytes > 0:
        max_total_kv_tokens = int((kv_cache_gib_per_gpu * BYTES_PER_GIB) // per_gpu_token_kv_bytes)
    else:
        max_total_kv_tokens = 0

    # 6. Max Concurrency at Target Context
    if target_context > 0 and max_total_kv_tokens > 0:
        max_concurrency_at_target_context = int(max_total_kv_tokens // target_context)
    else:
        max_concurrency_at_target_context = 0

    # 7. Recommended Max Model Length (never rounds positive block budgets down to 0)
    model_max_len = model_info["default_max_len"]
    if max_total_kv_tokens >= target_context:
        recommended_max_model_len = min(target_context, model_max_len)
    elif max_total_kv_tokens >= 1024:
        recommended_max_model_len = min((max_total_kv_tokens // 256) * 256, model_max_len)
    elif max_total_kv_tokens >= 16:
        recommended_max_model_len = min((max_total_kv_tokens // 16) * 16, model_max_len)
    else:
        recommended_max_model_len = 0

    # 8. TP Compatibility & Overall Feasibility
    tp_valid, tp_invalid_reason = _is_tp_valid_for_model(
        model_info, resolved_num_gpus, canonical_gpu_sku, vllm_quant
    )
    multinode_invalid = (
        resolved_num_gpus > sku_meta["max_gpus_per_node"]
        and math.gcd(resolved_num_gpus, sku_meta["max_gpus_per_node"]) < 2
    )

    fits_in_vram = (
        tp_valid
        and not multinode_invalid
        and (kv_cache_gib_per_gpu >= 0.5)
        and (max_total_kv_tokens >= target_context)
        and (recommended_max_model_len >= 16)
    )

    # 9. Diagnostics and Warnings
    warnings: List[str] = []

    if not tp_valid and tp_invalid_reason:
        warnings.append(f"TP Configuration Invalid: {tp_invalid_reason}")

    if multinode_invalid:
        warnings.append(
            f"Topology Invalid: {resolved_num_gpus} GPUs cannot be partitioned evenly across "
            f"{sku_meta['max_gpus_per_node']}-GPU worker nodes."
        )

    if not (kv_cache_gib_per_gpu >= 0.5 and max_total_kv_tokens >= target_context):
        tp_hint = (
            f"Minimum fitting tensor-parallel-size is {recommended_tp_size}."
            if recommended_tp_size is not None
            else f"No TP configuration up to 16x {canonical_gpu_sku} fits this precision and context."
        )
        ctx_hint = (
            f" Alternatively, lower --target-context to <= {recommended_max_model_len}."
            if kv_cache_gib_per_gpu >= 0.5 and recommended_max_model_len >= 16
            else ""
        )
        warnings.append(
            f"OOM Risk: Model weights ({weight_memory_per_gpu_gib:.2f} GiB/GPU) + CUDA overhead "
            f"({cuda_overhead_gib:.2f} GiB) leave insufficient KV-cache VRAM on "
            f"{resolved_num_gpus}x {canonical_gpu_sku} ({usable_vram_gib:.2f} GiB usable/GPU, "
            f"{max_total_kv_tokens:,} KV tokens vs {target_context:,} target). {tp_hint}{ctx_hint}"
        )
    elif fits_in_vram and max_concurrency_at_target_context < target_concurrency:
        tp_hint = f"TP={recommended_tp_size}" if recommended_tp_size else "a higher-VRAM SKU"
        warnings.append(
            f"KV-Cache Constrained: Maximum concurrency at context {target_context} is "
            f"{max_concurrency_at_target_context}, below target concurrency {target_concurrency}. "
            f"Consider {tp_hint}, FP8 KV cache, or lowering --max-model-len."
        )

    if resolved_num_gpus > 1 and kv_heads_per_gpu * resolved_num_gpus > model_info["num_kv_heads"]:
        rep_factor = (kv_heads_per_gpu * resolved_num_gpus) / model_info["num_kv_heads"]
        warnings.append(
            f"KV Head Replication: TP={resolved_num_gpus} exceeds num_kv_heads "
            f"({model_info['num_kv_heads']}); vLLM assigns {kv_heads_per_gpu} KV head(s) per worker "
            f"({rep_factor:.1f}x replication, {per_token_kv_bytes:,} aggregate bytes/token)."
        )

    if target_context > model_max_len:
        warnings.append(
            f"Context Length Exceeded: Target context {target_context} exceeds model default maximum "
            f"{model_max_len}. Ensure RoPE scaling / YaRN is configured before raising --max-model-len."
        )

    if vllm_quant == "fp8" and not sku_meta["supports_fp8_w8a8"]:
        warnings.append(
            f"Hardware Precision Warning: {canonical_gpu_sku} (SM{sku_meta['compute_capability'][0]}"
            f"{sku_meta['compute_capability'][1]}) lacks native W8A8 FP8 Tensor Cores (requires SM89+ "
            "Ada/Hopper). Unquantized checkpoints with --quantization=fp8 will fail on vLLM 0.6.4 unless "
            "serving a pre-quantized checkpoint supported by Marlin weight-only kernels."
        )

    if vllm_quant in ("awq", "gptq", "compressed-tensors"):
        warnings.append(
            f"Pre-Quantized Checkpoint Required: --quantization={vllm_quant} requires replacing "
            f"default catalog model '{model_info['hf_id']}' with a matching {vllm_quant.upper()} checkpoint."
        )

    if resolved_num_gpus > 1 and sku_meta["interconnect"] == "pcie":
        warnings.append(
            f"Interconnect Bandwidth Warning: {canonical_gpu_sku} lacks NVLink/NVSwitch; TP={resolved_num_gpus} "
            "all-reduce collectives traverse PCIe, increasing inter-token latency."
        )

    if resolved_num_gpus > 8 and fits_in_vram:
        warnings.append(
            f"Multinode Serving Required: TP={resolved_num_gpus} exceeds 8 GPUs/node; generated manifest "
            f"uses LeaderWorkerSet + Ray across {resolved_num_gpus // 8}x 8-GPU pods (requires RDMA/InfiniBand)."
        )

    # 10. Generate Validated vLLM 0.6.4 CLI Arguments & Manifest ONLY when Feasible
    vllm_cli_args: List[str] = []
    k8s_manifest_yaml: str = ""

    if fits_in_vram:
        max_num_seqs = max(1, min(target_concurrency, max_concurrency_at_target_context))
        vllm_cli_args = [
            f"--model={model_info['hf_id']}",
            f"--tensor-parallel-size={resolved_num_gpus}",
            f"--dtype={vllm_dtype}",
            f"--kv-cache-dtype={vllm_kv_dtype}",
            f"--gpu-memory-utilization={gpu_memory_utilization:.2f}",
            f"--max-model-len={recommended_max_model_len}",
            f"--max-num-seqs={max_num_seqs}",
        ]
        if vllm_quant is not None:
            vllm_cli_args.append(f"--quantization={vllm_quant}")
            if vllm_quant == "bitsandbytes":
                vllm_cli_args.append("--load-format=bitsandbytes")
        vllm_cli_args.extend(
            [
                "--enable-prefix-caching",
                "--disable-log-requests",
            ]
        )

        deployment_name = f"vllm-{canonical_model_key.replace('.', '-')}"
        k8s_manifest_yaml = _generate_k8s_deployment_manifest(
            deployment_name=deployment_name,
            hf_id=model_info["hf_id"],
            gpu_sku=canonical_gpu_sku,
            num_gpus=resolved_num_gpus,
            vllm_cli_args=vllm_cli_args,
        )

    result = VLLMTuningResult(
        model_key=canonical_model_key,
        hf_id=model_info["hf_id"],
        gpu_sku=canonical_gpu_sku,
        num_gpus=resolved_num_gpus,
        dtype=norm_dtype,
        kv_dtype=norm_kv_dtype,
        target_context=target_context,
        target_concurrency=target_concurrency,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_capacity_gib=gpu_capacity_gib,
        total_gpu_capacity_gib=total_gpu_capacity_gib,
        weight_memory_gib=weight_memory_gib,
        weight_memory_per_gpu_gib=weight_memory_per_gpu_gib,
        cuda_overhead_gib=cuda_overhead_gib,
        total_cuda_overhead_gib=total_cuda_overhead_gib,
        usable_vram_gib=usable_vram_gib,
        total_usable_vram_gib=total_usable_vram_gib,
        available_kv_cache_gib=available_kv_cache_gib,
        per_token_kv_bytes=per_token_kv_bytes,
        max_total_kv_tokens=max_total_kv_tokens,
        max_concurrency_at_target_context=max_concurrency_at_target_context,
        recommended_max_model_len=recommended_max_model_len,
        recommended_tp_size=recommended_tp_size,
        fits_in_vram=fits_in_vram,
        warnings=warnings,
        vllm_cli_args=vllm_cli_args,
        k8s_manifest_yaml=k8s_manifest_yaml,
    )

    return result.to_dict()


if __name__ == "__main__":
    import json

    res = calculate_vllm_tuning(
        model_key="llama-3.1-70b",
        gpu_sku="H100-80GB",
        dtype="fp8",
        kv_dtype="fp8",
        target_context=16384,
        target_concurrency=32,
    )
    print(json.dumps(res, indent=2))
