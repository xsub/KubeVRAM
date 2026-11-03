"""
KubeVRAM vLLM Resource & KV-Cache Sizing Calculator.

Deterministic calculations for:
1. Exact model weight footprint based on precision (FP16, FP8, INT4).
2. Per-token KV cache memory consumption (GQA-aware).
3. Usable VRAM after CUDA context and CUDA Graph capture overhead.
4. Maximum sequence concurrency at target context length.
5. Recommended Tensor Parallelism (TP) size and vLLM CLI flags.
6. Ready-to-apply Kubernetes Deployment manifest.
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
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "llama-3.1-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "params_b": 70.6,
        "num_layers": 80,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "llama-3.1-405b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "params_b": 405.0,
        "num_layers": 126,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.24,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 32768,
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.5,
        "num_layers": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "deepseek-r1-32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "params_b": 32.5,
        "num_layers": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_max_len": 131072,
    },
    "mixtral-8x22b": {
        "hf_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "params_b": 141.0,
        "num_layers": 56,
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

# Precision to bytes mapping
DTYPE_BYTES_MAP: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5,
    "gptq": 0.5,
    "int4": 0.5,
}

# KV Cache precision to bytes mapping
KV_DTYPE_BYTES_MAP: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

CUDA_CONTEXT_GRAPH_OVERHEAD_GIB: float = 2.8


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
    recommended_tp_size: int
    fits_in_vram: bool
    warnings: List[str] = field(default_factory=list)
    vllm_cli_args: List[str] = field(default_factory=list)
    k8s_manifest_yaml: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary matching KubeVRAM standard schema."""
        data = asdict(self)
        # Aliases for backward/generic compatibility
        data["weight_memory_gb"] = self.weight_memory_gib
        data["kv_cache_gib"] = self.available_kv_cache_gib
        data["cuda_context_overhead_gib"] = self.cuda_overhead_gib
        return data


def _resolve_model(model_key: str) -> Tuple[str, Dict[str, Any]]:
    """Resolves a model key or case-insensitive alias to catalog entry."""
    norm_key = model_key.strip().lower()
    if norm_key in MODEL_CATALOG:
        return norm_key, MODEL_CATALOG[norm_key]

    # Check for prefix or substring matches
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

    # Case-insensitive search
    for k, v in GPU_CATALOG.items():
        if k.lower() == norm_sku.lower():
            return k, v

    # Partial match aliases
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


def _get_gpu_memory_utilization(gpu_capacity_gib: float) -> float:
    """
    Returns recommended vLLM --gpu-memory-utilization factor.
    Leaves 2.5-3.5 GB headroom for PyTorch NCCL buffers, CUDA context, and CUDA graphs.
    """
    if gpu_capacity_gib >= 40.0:
        return 0.92
    elif gpu_capacity_gib <= 24.0:
        return 0.88
    return 0.90


def _calculate_optimal_tp(
    weight_memory_gib: float,
    gpu_capacity_gib: float,
    per_token_kv_bytes: int,
    target_context: int,
    target_concurrency: int,
) -> int:
    """Calculates recommended Tensor Parallelism (TP) size (powers of 2: 1, 2, 4, 8, 16)."""
    candidate_tps = [1, 2, 4, 8, 16]
    utilization = _get_gpu_memory_utilization(gpu_capacity_gib)
    usable_per_gpu = gpu_capacity_gib * utilization

    # First pass: find smallest TP that fits weights + overhead with enough KV cache for target context
    min_fitting_tp: Optional[int] = None
    target_tokens = target_context * max(1, target_concurrency)

    for tp in candidate_tps:
        weight_per_gpu = weight_memory_gib / tp
        kv_cache_per_gpu = usable_per_gpu - weight_per_gpu - CUDA_CONTEXT_GRAPH_OVERHEAD_GIB
        if kv_cache_per_gpu > 0.5:
            total_kv_bytes = kv_cache_per_gpu * tp * (1024**3)
            total_tokens = int(total_kv_bytes / per_token_kv_bytes)
            if total_tokens >= target_context:
                if min_fitting_tp is None:
                    min_fitting_tp = tp
                if total_tokens >= target_tokens:
                    return tp

    if min_fitting_tp is not None:
        return min_fitting_tp
    return 8


def _generate_k8s_deployment_manifest(
    deployment_name: str,
    hf_id: str,
    gpu_sku: str,
    num_gpus: int,
    vllm_cli_args: List[str],
) -> str:
    """Generates ready-to-apply Kubernetes Deployment manifest with NVIDIA runtime and /dev/shm."""
    if gpu_sku.upper().startswith("MIG-"):
        gpu_resource_name = f"nvidia.com/{gpu_sku.lower()}"
    else:
        gpu_resource_name = "nvidia.com/gpu"

    cpu_req = max(8, 4 * num_gpus)
    cpu_lim = max(16, 8 * num_gpus)
    mem_req_gi = max(32, 16 * num_gpus)
    mem_lim_gi = max(64, 32 * num_gpus)

    args_formatted = "\n".join(f"            - \"{arg}\"" for arg in vllm_cli_args)

    return f"""apiVersion: apps/v1
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
      labels:
        app: {deployment_name}
    spec:
      runtimeClassName: nvidia
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.4
          args:
{args_formatted}
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


def calculate_vllm_tuning(
    model_key: str = "llama-3.1-70b",
    gpu_sku: str = "H100-80GB",
    num_gpus: int = 1,
    dtype: str = "fp8",
    kv_dtype: str = "fp8",
    target_context: int = 16384,
    target_concurrency: int = 32,
) -> Dict[str, Any]:
    """
    Computes exact VRAM footprints, KV cache token limits, concurrency capacity,
    vLLM CLI flags, and Kubernetes manifests for model serving.
    """
    canonical_model_key, model_info = _resolve_model(model_key)
    canonical_gpu_sku, gpu_mib = _resolve_gpu_sku(gpu_sku)

    num_gpus = max(1, int(num_gpus))
    target_context = max(512, int(target_context))
    target_concurrency = max(1, int(target_concurrency))

    norm_dtype = dtype.strip().lower()
    bytes_per_param = DTYPE_BYTES_MAP.get(norm_dtype, 2.0 if "16" in norm_dtype else 1.0)

    norm_kv_dtype = kv_dtype.strip().lower()
    kv_bytes = KV_DTYPE_BYTES_MAP.get(norm_kv_dtype, 1.0 if "8" in norm_kv_dtype else 2.0)

    # 1. Exact Model Weight Footprint
    # VRAM_weights = params_b * bytes_per_param * 1.04 GiB (accounting for alignment and overhead)
    weight_memory_gib = round(model_info["params_b"] * bytes_per_param * 1.04, 3)
    weight_memory_per_gpu_gib = round(weight_memory_gib / num_gpus, 3)

    # 2. CUDA Context and CUDA Graph Overhead
    cuda_overhead_gib = CUDA_CONTEXT_GRAPH_OVERHEAD_GIB
    total_cuda_overhead_gib = round(cuda_overhead_gib * num_gpus, 3)

    # 3. GPU Capacity & Usable VRAM
    gpu_capacity_gib = round(gpu_mib / 1024.0, 3)
    total_gpu_capacity_gib = round(gpu_capacity_gib * num_gpus, 3)
    gpu_memory_utilization = _get_gpu_memory_utilization(gpu_capacity_gib)
    usable_vram_gib = round(gpu_capacity_gib * gpu_memory_utilization, 3)
    total_usable_vram_gib = round(usable_vram_gib * num_gpus, 3)

    # 4. Available KV Cache Memory
    kv_cache_gib_per_gpu = usable_vram_gib - weight_memory_per_gpu_gib - cuda_overhead_gib
    if kv_cache_gib_per_gpu > 0:
        available_kv_cache_gib = round(kv_cache_gib_per_gpu * num_gpus, 3)
    else:
        available_kv_cache_gib = 0.0

    # 5. Per-Token KV Cache Bytes (Grouped-Query Attention aware)
    # Bytes_KV/token = 2 * num_layers * num_kv_heads * head_dim * kv_bytes
    per_token_kv_bytes = int(
        2 * model_info["num_layers"] * model_info["num_kv_heads"] * model_info["head_dim"] * kv_bytes
    )

    # 6. Max KV Tokens Across GPUs
    if available_kv_cache_gib > 0 and per_token_kv_bytes > 0:
        total_kv_bytes = available_kv_cache_gib * (1024**3)
        max_total_kv_tokens = int(total_kv_bytes / per_token_kv_bytes)
    else:
        max_total_kv_tokens = 0

    # 7. Max Concurrency at Target Context
    if target_context > 0 and max_total_kv_tokens > 0:
        max_concurrency_at_target_context = int(max_total_kv_tokens / target_context)
    else:
        max_concurrency_at_target_context = 0

    # 8. Recommended Max Model Length
    if max_total_kv_tokens >= target_context:
        recommended_max_model_len = min(target_context, model_info["default_max_len"])
    elif max_total_kv_tokens > 512:
        # Snap to nearest multiple of 1024
        recommended_max_model_len = (max_total_kv_tokens // 1024) * 1024
    else:
        recommended_max_model_len = min(target_context, model_info["default_max_len"])

    # 9. Feasibility & Recommended TP Size
    fits_in_vram = (kv_cache_gib_per_gpu > 0.5) and (max_total_kv_tokens >= target_context)

    recommended_tp_size = _calculate_optimal_tp(
        weight_memory_gib=weight_memory_gib,
        gpu_capacity_gib=gpu_capacity_gib,
        per_token_kv_bytes=per_token_kv_bytes,
        target_context=target_context,
        target_concurrency=target_concurrency,
    )

    # 10. Diagnostics and Warnings
    warnings: List[str] = []

    if not fits_in_vram:
        req_vram_per_gpu = weight_memory_per_gpu_gib + cuda_overhead_gib
        warnings.append(
            f"OOM Risk: Model weights ({weight_memory_per_gpu_gib:.2f} GiB/GPU) + CUDA overhead "
            f"({cuda_overhead_gib:.2f} GiB) exceed or leave insufficient usable VRAM "
            f"({usable_vram_gib:.2f} GiB) on {num_gpus}x {canonical_gpu_sku}. "
            f"Recommended tensor-parallel-size is {recommended_tp_size}."
        )
    elif max_concurrency_at_target_context < target_concurrency:
        warnings.append(
            f"KV-Cache Constrained: Maximum concurrency at context {target_context} is "
            f"{max_concurrency_at_target_context}, which is lower than target concurrency {target_concurrency}. "
            f"Consider increasing num_gpus (TP={recommended_tp_size}), using FP8 KV cache, or lowering max-model-len."
        )

    if canonical_gpu_sku.startswith("MIG-") and num_gpus > 1:
        warnings.append(
            f"MIG Multi-GPU Limitation: Running Tensor Parallelism across {num_gpus} MIG slices "
            f"disables NVLink P2P communication. Communication will fall back to PCIe/host memory."
        )

    if target_context > model_info["default_max_len"]:
        warnings.append(
            f"Context Length Exceeded: Target context {target_context} exceeds model default maximum "
            f"{model_info['default_max_len']}. Ensure RoPE scaling or Yarn is enabled."
        )

    if num_gpus > 1 and model_info["num_kv_heads"] % num_gpus != 0:
        warnings.append(
            f"KV Head Divisibility: Number of GPUs ({num_gpus}) does not evenly divide num_kv_heads "
            f"({model_info['num_kv_heads']}). vLLM will replicate KV heads across TP workers."
        )

    # 11. Generate vLLM CLI Arguments
    effective_tp = num_gpus if fits_in_vram else recommended_tp_size
    vllm_cli_args = [
        f"--model={model_info['hf_id']}",
        f"--tensor-parallel-size={effective_tp}",
        f"--dtype={norm_dtype}",
        f"--kv-cache-dtype={norm_kv_dtype}",
        f"--gpu-memory-utilization={gpu_memory_utilization:.2f}",
        f"--max-model-len={recommended_max_model_len}",
        f"--max-num-seqs={max(1, min(target_concurrency, max(1, max_concurrency_at_target_context)))}",
    ]
    if norm_dtype in ("fp8", "int8", "awq", "gptq"):
        vllm_cli_args.append(f"--quantization={norm_dtype}")
    vllm_cli_args.extend([
        "--enable-prefix-caching",
        "--disable-log-requests",
    ])

    # 12. Generate Kubernetes Deployment YAML
    deployment_name = f"vllm-{canonical_model_key.replace('.', '-')}"
    k8s_manifest_yaml = _generate_k8s_deployment_manifest(
        deployment_name=deployment_name,
        hf_id=model_info["hf_id"],
        gpu_sku=canonical_gpu_sku,
        num_gpus=effective_gpus if "effective_gpus" in locals() else effective_tp,
        vllm_cli_args=vllm_cli_args,
    )

    result = VLLMTuningResult(
        model_key=canonical_model_key,
        hf_id=model_info["hf_id"],
        gpu_sku=canonical_gpu_sku,
        num_gpus=num_gpus,
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
    import sys

    # Quick test execution
    res = calculate_vllm_tuning(
        model_key="llama-3.1-70b",
        gpu_sku="H100-80GB",
        num_gpus=1,
        dtype="fp8",
        kv_dtype="fp8",
        target_context=16384,
        target_concurrency=32,
    )
    print(json.dumps(res, indent=2))
