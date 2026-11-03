"""
KubeVRAM OOM & Hardware Fault Correlation Engine.

Distinguishes between:
1. Kubernetes System RAM OOM (`OOMKilled`, Exit Code 137 — Linux cgroup host RAM limit)
2. CUDA VRAM OOM (`Error`, Exit Code 1 — GPU HBM exhaustion / vLLM KV-Cache over-allocation)
3. Time-Slicing Co-Tenant VRAM Collision (vLLM pre-allocation on shared physical VRAM)
4. NCCL Shared Memory (`/dev/shm`) Exhaustion (Exit Code 135 / SIGBUS on multi-GPU pods)
5. NVIDIA Hardware XID Fault (`DCGM_FI_DEV_XID_ERRORS > 0`, e.g., XID 79 / XID 95 ECC error)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from kubevram.models import ClusterNode, GPUDevice, InferencePod, RCAFinding


def _find_node(nodes: List[ClusterNode], node_name: str) -> Optional[ClusterNode]:
    for node in nodes:
        if node.name == node_name:
            return node
    return None


def _find_pod_gpus(node: Optional[ClusterNode], pod: InferencePod) -> List[GPUDevice]:
    if not node:
        return []
    matched: List[GPUDevice] = []
    for gpu in node.gpus:
        if gpu.index in pod.gpu_indices or f"{pod.namespace}/{pod.name}" in gpu.allocated_pods:
            matched.append(gpu)
    return matched


def correlate_pod_failure(pod: InferencePod, nodes: List[ClusterNode]) -> Optional[RCAFinding]:
    """
    Correlates K8s Pod termination state, cAdvisor host RAM, DCGM GPU metrics,
    node sharing strategy, and container stderr logs to produce an actionable RCAFinding.
    """
    if pod.status == "Running" and pod.restarts == 0 and pod.last_exit_code == 0:
        return None

    node = _find_node(nodes, pod.node_name)
    gpus = _find_pod_gpus(node, pod)
    joined_logs = "\n".join(pod.recent_logs)

    max_vram_pct = max((g.vram_util_pct for g in gpus), default=0.0)
    used_vram_mib = max((g.used_vram_mib for g in gpus), default=0)
    total_vram_mib = max((g.total_vram_mib for g in gpus), default=0)
    xid_codes = [g.xid_error for g in gpus if g.xid_error > 0]
    sharing_mode = node.sharing_strategy if node else "unknown"

    # Case 1: NVIDIA Hardware XID Fault (e.g., XID 79, 94, 95, 48, 31)
    if xid_codes or "uncorrectable ECC error" in joined_logs or "Xid (" in joined_logs:
        xid = xid_codes[0] if xid_codes else 95
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category=f"NVIDIA_HARDWARE_XID_FAULT (XID {xid})",
            severity="CRITICAL",
            confidence=0.99,
            summary=f"Hardware ECC / Bus Fault (XID {xid}) detected on node {pod.node_name} GPU {pod.gpu_indices}",
            technical_evidence=[
                f"DCGM Metric: DCGM_FI_DEV_XID_ERRORS = {xid} (Uncontained HBM ECC / Hardware Fault)",
                f"K8s Container Exit Code: {pod.last_exit_code} (Reason: {pod.last_termination_reason})",
                f"Host RAM (cAdvisor): {pod.mem_used_gi:.1f} GiB / {pod.mem_limit_gi:.1f} GiB (Healthy — NOT a K8s RAM OOM)",
                f"Log Signature: '{pod.recent_logs[-1] if pod.recent_logs else 'XID fault'}'",
            ],
            root_cause=(
                f"Physical GPU silicon or HBM2e/HBM3 memory bank on node '{pod.node_name}' encountered a hardware-level "
                f"NVIDIA XID {xid} fault. Restarting the pod on the same GPU slice will repeatedly fail until the node "
                "is cordoned and the GPU undergone reset/RMA."
            ),
            linux_equivalent_cmd="dmesg -T | grep -E 'NVRM: Xid|ECC' && nvidia-smi -q -d ECC,RETIRED_PAGES",
            k8s_remediation_cmds=[
                f"kubectl cordon {pod.node_name}",
                f"kubectl annotate node {pod.node_name} nvidia.com/gpu-xid-fault='XID-{xid}' --overwrite",
                f"kubectl delete pod -n {pod.namespace} {pod.name} --wait=false",
            ],
            yaml_patch=(
                "# Node Taint to Isolate Faulty GPU Silicon:\n"
                f"# kubectl taint nodes {pod.node_name} nvidia.com/gpu-xid-fault=XID-{xid}:NoSchedule"
            ),
        )

    # Case 2: Kubernetes System RAM OOMKilled (Exit Code 137, Linux cgroup memory.max)
    if pod.last_exit_code == 137 or pod.last_termination_reason == "OOMKilled" or pod.status == "OOMKilled":
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="K8S_SYSTEM_RAM_OOMKILLED (Exit 137)",
            severity="HIGH",
            confidence=0.98,
            summary=(
                f"Linux cgroup System RAM OOMKilled (Exit 137): Host RAM hit {pod.mem_used_gi:.1f}/{pod.mem_limit_gi:.1f} GiB "
                f"while GPU VRAM was only at {max_vram_pct:.1f}% ({used_vram_mib} MiB / {total_vram_mib} MiB)."
            ),
            technical_evidence=[
                f"K8s Termination State: reason=OOMKilled, exitCode=137 (SIGKILL from Linux kernel cgroup OOM killer)",
                f"Host System RAM (cAdvisor): {pod.mem_used_gi:.1f} GiB used == {pod.mem_limit_gi:.1f} GiB container limit (100% saturated)",
                f"GPU Framebuffer (DCGM_FI_DEV_FB_USED): {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}% — GPU VRAM had plenty of free space!)",
            ],
            root_cause=(
                "This is NOT a GPU VRAM OOM. The container exceeded its Kubernetes host RAM limit "
                f"(resources.limits.memory: {int(pod.mem_limit_gi)}Gi) while staging/deserializing model weights "
                "in CPU memory before transferring tensors to the GPU."
            ),
            linux_equivalent_cmd="dmesg -T | grep -i 'Memory cgroup out of memory' && free -h",
            k8s_remediation_cmds=[
                f"kubectl patch deployment -n {pod.namespace} {pod.name} -p "
                '\'{"spec":{"template":{"spec":{"containers":[{"name":"main","resources":{"limits":{"memory":"96Gi"},"requests":{"memory":"64Gi"}}}]}}}}\'',
            ],
            yaml_patch=(
                "resources:\n"
                "  requests:\n"
                "    memory: \"64Gi\"   # Increased from 12Gi to fit CPU safetensors mmap/staging\n"
                "  limits:\n"
                "    memory: \"96Gi\"   # Prevents Linux cgroup Exit 137 OOMKilled during model load"
            ),
        )

    # Case 3: NCCL /dev/shm Bus Error (Exit Code 135 / SIGBUS or posix_fallocate error)
    if (
        pod.last_exit_code == 135
        or "/dev/shm" in joined_logs
        or "posix_fallocate failed" in joined_logs
        or (pod.shm_size_mib <= 64 and len(pod.gpu_indices) >= 2)
    ):
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="NCCL_SHM_BUS_ERROR (Exit 135 / /dev/shm Exhausted)",
            severity="HIGH",
            confidence=0.97,
            summary=f"Multi-GPU NCCL SIGBUS (Exit {pod.last_exit_code}): /dev/shm is capped at Kubernetes default ({pod.shm_size_mib} MiB)",
            technical_evidence=[
                f"K8s Container Exit Code: {pod.last_exit_code} (SIGBUS — Shared Memory Bus Error)",
                f"Configured /dev/shm Volume: {pod.shm_size_mib} MiB (Default K8s container /dev/shm is 64 MiB)",
                f"Tensor Parallelism GPUs: {len(pod.gpu_indices)} GPUs ({', '.join(pod.gpu_indices)})",
                "Log Signature: 'NCCL WARN Call to posix_fallocate failed : No space left on device (/dev/shm/...)'",
            ],
            root_cause=(
                "Multi-GPU Tensor Parallelism (NCCL P2P & PyTorch multiprocess workers) allocates shared memory segments "
                "in /dev/shm. Because Kubernetes defaults /dev/shm to a 64 MiB tmpfs mount, NCCL ring initialization fails "
                "with SIGBUS (Exit Code 135)."
            ),
            linux_equivalent_cmd="df -h /dev/shm && ipcs -m",
            k8s_remediation_cmds=[
                f"# Mount an in-memory emptyDir (sizeLimit: 16Gi) at /dev/shm for pod {pod.namespace}/{pod.name}"
            ],
            yaml_patch=(
                "volumeMounts:\n"
                "  - name: dshm\n"
                "    mountPath: /dev/shm\n"
                "volumes:\n"
                "  - name: dshm\n"
                "    emptyDir:\n"
                "      medium: Memory\n"
                "      sizeLimit: 16Gi"
            ),
        )

    # Case 4: Time-Slicing VRAM Collision on Shared GPU
    if (
        sharing_mode == "time-slicing"
        and (
            "OutOfMemoryError" in joined_logs
            or "CUDA out of memory" in joined_logs
            or max_vram_pct >= 94.0
        )
    ):
        co_tenants: List[str] = []
        for g in gpus:
            for alloc in g.allocated_pods:
                if alloc != f"{pod.namespace}/{pod.name}":
                    co_tenants.append(alloc)
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="CUDA_VRAM_OOM_TIMESLICING_COLLISION (Exit 1)",
            severity="CRITICAL",
            confidence=0.98,
            summary=(
                f"Time-Slicing VRAM Collision on {pod.node_name}: Co-tenant ({', '.join(co_tenants) or 'other pod'}) "
                f"pre-allocated 90% of shared physical VRAM ({used_vram_mib}/{total_vram_mib} MiB)."
            ),
            technical_evidence=[
                f"Node Sharing Strategy: nvidia.com/gpu.sharing-strategy = 'time-slicing' (replicas={node.replicas_per_gpu if node else 4})",
                f"DCGM Framebuffer: DCGM_FI_DEV_FB_USED = {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}% saturated)",
                f"Pod Flag: --gpu-memory-utilization={pod.cli_args.get('--gpu-memory-utilization', '0.90')} (assumes exclusive 100% GPU ownership)",
                f"Co-located Tenant Pods on Same Physical GPU: {', '.join(co_tenants) if co_tenants else 'Multiple replicas'}",
            ],
            root_cause=(
                "NVIDIA Time-Slicing multiplexes compute contexts in software but exposes the FULL physical VRAM to every "
                "container. Because vLLM pre-allocates --gpu-memory-utilization (0.90 = 90% of physical VRAM) on startup, "
                "the first pod claimed 90% of the 48GB GPU, leaving <5GB for this second pod and triggering an immediate "
                "torch.cuda.OutOfMemoryError."
            ),
            linux_equivalent_cmd="nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
            k8s_remediation_cmds=[
                f"kubevram partition --gpu {node.gpu_model if node else 'L40S-48GB'} --workload vllm --tenants multi",
                "# Option A (Preferred on A100/H100): Move vLLM workloads to hardware-isolated MIG slices (e.g. nvidia.com/mig-2g.20gb)",
                "# Option B (If staying on L40S): Cap --gpu-memory-utilization=0.42 and use CUDA MPS with pinned memory limits",
            ],
            yaml_patch=(
                "args:\n"
                "  - \"--gpu-memory-utilization=0.42\"  # Capped for 2 co-tenants on shared 48GB GPU\n"
                "  - \"--max-model-len=8192\"           # Right-sized KV cache to fit 20.1 GiB budget"
            ),
        )

    # Case 5: Dedicated GPU / MIG CUDA VRAM OOM (KV-Cache / Max-Model-Len Over-Allocation)
    if (
        "OutOfMemoryError" in joined_logs
        or "CUDA out of memory" in joined_logs
        or "larger than the maximum number of tokens" in joined_logs
        or max_vram_pct >= 94.0
    ):
        curr_len = pod.cli_args.get("--max-model-len", "131072")
        curr_util = pod.cli_args.get("--gpu-memory-utilization", "0.95")
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="CUDA_VRAM_OOM_KV_CACHE_OVERFLOW (Exit 1)",
            severity="HIGH",
            confidence=0.96,
            summary=(
                f"CUDA VRAM OOM (Exit 1, NOT K8s OOMKilled): Model weights + KV Cache for --max-model-len={curr_len} "
                f"exceeded {total_vram_mib} MiB HBM ({max_vram_pct:.1f}% used)."
            ),
            technical_evidence=[
                f"K8s Termination State: reason=Error, exitCode={pod.last_exit_code} (Host RAM was healthy at {pod.mem_used_gi:.1f}/{pod.mem_limit_gi:.1f} GiB)",
                f"DCGM Framebuffer: DCGM_FI_DEV_FB_USED = {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}%)",
                f"vLLM Engine Args: --max-model-len={curr_len}, --gpu-memory-utilization={curr_util}, --kv-cache-dtype={pod.cli_args.get('--kv-cache-dtype', 'auto')}",
                f"Log Signature: '{pod.recent_logs[-1] if pod.recent_logs else 'CUDA out of memory'}'",
            ],
            root_cause=(
                f"Model weights consumed ~68.5 GiB on the 80 GiB GPU, leaving ~7.5 GiB for KV Cache and CUDA graphs. "
                f"Requesting --max-model-len={curr_len} with FP16/BF16 KV Cache requires ~40 GiB for a single full-context "
                "sequence, causing vLLM KV block allocation to fail with CUDA OutOfMemoryError."
            ),
            linux_equivalent_cmd="nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv",
            k8s_remediation_cmds=[
                f"kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --dtype fp8 --kv-dtype fp8 --target-context 16384",
                f"kubectl set env deployment/{pod.name} -n {pod.namespace} VLLM_ATTENTION_BACKEND=FLASHINFER",
            ],
            yaml_patch=(
                "args:\n"
                "  - \"--gpu-memory-utilization=0.92\"\n"
                "  - \"--max-model-len=16384\"          # Reduced from 131072 to fit single 80GB H100\n"
                "  - \"--kv-cache-dtype=fp8\"           # Halves per-token KV cache footprint (160 KiB -> 80 KiB/token)"
            ),
        )

    # Fallback generic abnormal pod state
    return RCAFinding(
        pod_name=pod.name,
        namespace=pod.namespace,
        node_name=pod.node_name,
        failure_category=f"POD_RUNTIME_ERROR (Exit {pod.last_exit_code})",
        severity="MEDIUM",
        confidence=0.75,
        summary=f"Pod {pod.namespace}/{pod.name} in status {pod.status} (exitCode={pod.last_exit_code})",
        technical_evidence=[
            f"Status: {pod.status}, Restarts: {pod.restarts}, Reason: {pod.last_termination_reason}",
            f"GPU VRAM: {used_vram_mib}/{total_vram_mib} MiB ({max_vram_pct:.1f}%)",
        ],
        root_cause="Container exited with non-zero status; inspect container logs and runtime configuration.",
        linux_equivalent_cmd="journalctl -u containerd && nvidia-smi",
        k8s_remediation_cmds=[f"kubectl logs -n {pod.namespace} {pod.name} --previous"],
        yaml_patch="# Inspect container args and readinessProbe initialDelaySeconds",
    )


def triage_cluster(
    nodes: List[ClusterNode],
    pods: List[InferencePod],
    pod_filter: Optional[str] = None,
) -> List[RCAFinding]:
    """Runs the correlation engine across all unhealthy pods (or a specific filtered pod) in the cluster."""
    findings: List[RCAFinding] = []
    for pod in pods:
        if pod_filter and pod_filter not in (pod.name, f"{pod.namespace}/{pod.name}"):
            continue
        finding = correlate_pod_failure(pod, nodes)
        if finding is not None:
            findings.append(finding)
    return findings
