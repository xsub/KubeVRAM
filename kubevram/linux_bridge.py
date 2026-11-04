"""
Linux to Kubernetes & DCGM Engineering Bridge for KubeVRAM.

Translates Linux/CUDA system administration commands, kernel debugging tools,
and mental models into Kubernetes + DCGM + KubeVRAM equivalents.
"""

from __future__ import annotations

from typing import Dict, List, Optional

BRIDGE_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "id": "gpu-list",
        "category": "Telemetry",
        "linux_cmd": "nvidia-smi",
        "linux_intent": "Query real-time GPU inventory, driver version, CUDA version, temperature, power draw, and overall utilization.",
        "k8s_equivalent": "kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.'nvidia\\.com/gpu'",
        "dcgm_promql": "DCGM_FI_DEV_GPU_UTIL{modelName=~'.*'}",
        "kubevram_cmd": "kubevram topology",
        "explanation": "On bare-metal Linux, nvidia-smi talks directly to the NVML driver library. In Kubernetes, the NVIDIA GPU Operator advertises GPUs as node extended resources (nvidia.com/gpu), while DCGM Exporter exposes real-time telemetry to Prometheus.",
    },
    {
        "id": "vram-usage",
        "category": "Telemetry",
        "linux_cmd": "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv",
        "linux_intent": "Inspect physical High Bandwidth Memory (HBM/VRAM) allocation and free space per GPU device.",
        "k8s_equivalent": "kubectl describe nodes | grep -A 8 'Allocated resources' | grep -iE 'nvidia.com/gpu'",
        "dcgm_promql": "DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)",
        "kubevram_cmd": "kubevram topology",
        "explanation": "The Kubernetes scheduler allocates GPUs as integer units, blind to internal VRAM consumption. DCGM framebuffer metrics (DCGM_FI_DEV_FB_USED) are critical to track actual memory pressure and prevent co-tenant OOM collisions.",
    },
    {
        "id": "gpu-topology",
        "category": "Topology",
        "linux_cmd": "nvidia-smi topo -m",
        "linux_intent": "Inspect NVLink, NVSwitch, and PCIe P2P interconnect fabric between GPUs for distributed LLM tensor parallelism.",
        "k8s_equivalent": "kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\"\\t\"}{.metadata.labels.nvidia\\.com/gpu\\.family}{\"\\t\"}{.metadata.labels.nvidia\\.com/gpu\\.product}{\"\\n\"}{end}'",
        "dcgm_promql": "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
        "kubevram_cmd": "kubevram topology --fabric",
        "explanation": "Tensor-parallel inference (vLLM/Triton TP>=2) requires high-bandwidth NVLink/NVSwitch meshes (900 GB/s) rather than PCIe bus traversal. In Kubernetes, node labels and device plugins expose topology to ensure co-located scheduling.",
    },
    {
        "id": "xid-dmesg",
        "category": "OOM & Faults",
        "linux_cmd": "dmesg -T | grep -iE 'xid|nvrm'",
        "linux_intent": "Detect NVIDIA driver kernel errors, hardware uncontained ECC errors, thermal throttles, and GPU page faults.",
        "k8s_equivalent": "kubectl get events -A --field-selector reason=NodeNotReady && kubectl logs -n gpu-operator -l app=nvidia-driver-daemonset --tail=100",
        "dcgm_promql": "DCGM_FI_DEV_XID_ERRORS > 0",
        "kubevram_cmd": "kubevram triage",
        "explanation": "Linux dmesg logs NVRM Xid codes (e.g., Xid 95 for uncontained ECC memory bank errors, Xid 79 for GPU fallen off bus, Xid 31 for MMU page faults). DCGM tracks DCGM_FI_DEV_XID_ERRORS to drive triage and node isolation.",
    },
    {
        "id": "host-oom",
        "category": "OOM & Faults",
        "linux_cmd": "dmesg -T | grep -i 'out of memory' && free -h",
        "linux_intent": "Check if the Linux kernel OOM Killer terminated processes due to host system RAM exhaustion.",
        "k8s_equivalent": "kubectl get pods -A -o jsonpath='{range .items[?(@.status.containerStatuses[*].lastState.terminated.reason==\"OOMKilled\")]}{.metadata.namespace}/{.metadata.name}: {.status.containerStatuses[*].lastState.terminated.exitCode}{\"\\n\"}{end}'",
        "dcgm_promql": "container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.95",
        "kubevram_cmd": "kubevram triage",
        "explanation": "Pods terminated with status OOMKilled and exit code 137 are killed by the Linux cgroup host memory subsystem, distinctly different from PyTorch CUDA OutOfMemoryError, which exhausts GPU HBM VRAM inside the CUDA context.",
    },
    {
        "id": "shm-nccl",
        "category": "IPC & Memory",
        "linux_cmd": "df -h /dev/shm && ipcs -m",
        "linux_intent": "Inspect POSIX shared memory allocations and IPC segments used by multi-process tensor parallelism.",
        "k8s_equivalent": "kubectl describe pod <pod-name> | grep -A 4 -i 'Volumes:' | grep -i 'emptyDir.*Memory'",
        "dcgm_promql": "container_fs_usage_bytes{container=\"\", device=~'.*shm.*'}",
        "kubevram_cmd": "kubevram rbac-audit",
        "explanation": "Kubernetes containers default to a 64 MiB /dev/shm mount. Distributed PyTorch/vLLM multi-GPU communication (NCCL) relies on shared memory buffers for IPC; failing to mount an emptyDir medium=Memory causes sudden SIGBUS crashes (Exit Code 135).",
    },
    {
        "id": "numa-pinning",
        "category": "Topology",
        "linux_cmd": "numactl --hardware && taskset -p <pid>",
        "linux_intent": "Verify CPU core affinity, GPU socket affinity, and avoid cross-NUMA bus latency penalties for model staging.",
        "k8s_equivalent": "kubectl get node <node-name> -o jsonpath='{.metadata.labels.nvidia\\.com/gpu\\.numa}'",
        "dcgm_promql": "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
        "kubevram_cmd": "kubevram topology --fabric",
        "explanation": "The Kubernetes Topology Manager with numa-policy=single-numa-node coordinates CPU and GPU device allocation on the same NUMA node, avoiding cross-socket QPI/UPI bus saturation during large safetensors weight loading.",
    },
    {
        "id": "gpu-processes",
        "category": "Telemetry",
        "linux_cmd": "nvidia-smi pmon -s m -c 1 && ps -ef | grep vllm",
        "linux_intent": "List which Linux PIDs are running on which GPU and their exact VRAM footprint.",
        "k8s_equivalent": "kubectl top pods -A --containers",
        "dcgm_promql": "DCGM_FI_DEV_FB_USED",
        "kubevram_cmd": "kubevram topology",
        "explanation": "DCGM Exporter labels per-GPU metrics with exported_namespace and exported_pod so KubeVRAM maps physical GPU framebuffer utilization directly to Kubernetes Pods and owners.",
    },
    {
        "id": "mig-config",
        "category": "Partitioning",
        "linux_cmd": "nvidia-smi mig -lgi && nvidia-smi mig -lgi -i 0",
        "linux_intent": "Inspect Multi-Instance GPU (MIG) hardware-sliced GPU instances, compute instances, and profile allocations.",
        "k8s_equivalent": "kubectl get nodes -o custom-columns=NAME:.metadata.name,MIG_3G:.status.allocatable.'nvidia\\.com/mig-3g\\.40gb',MIG_2G:.status.allocatable.'nvidia\\.com/mig-2g\\.20gb'",
        "dcgm_promql": "DCGM_FI_DEV_MIG_MAX_SLICES",
        "kubevram_cmd": "kubevram partition --gpu A100-80GB --workload vllm --tenants multi",
        "explanation": "NVIDIA MIG partitions physical A100/H100/H200/A30 GPUs into isolated hardware instances with dedicated SMs, memory controllers, and crossbars. The NVIDIA GPU Operator registers these slices as distinct Kubernetes extended resources.",
    },
    {
        "id": "driver-status",
        "category": "Drivers",
        "linux_cmd": "lsmod | grep nvidia && modinfo nvidia",
        "linux_intent": "Check if NVIDIA kernel modules (nvidia, nvidia_uvm, nvidia_modeset) are loaded and verify driver kernel versions.",
        "k8s_equivalent": "kubectl get pods -n gpu-operator -l app.kubernetes.io/component=nvidia-driver -o wide",
        "dcgm_promql": "DCGM_FI_DRIVER_VERSION",
        "kubevram_cmd": "kubevram overview",
        "explanation": "In Kubernetes clusters, NVIDIA kernel modules are managed declaratively via NVIDIA GPU Operator Driver DaemonSets, eliminating host kernel driver drift and manual DKMS rebuilds during node OS upgrades.",
    },
    {
        "id": "mps-control",
        "category": "Partitioning",
        "linux_cmd": "ps -ef | grep nvidia-cuda-mps-control && echo quit | nvidia-cuda-mps-control",
        "linux_intent": "Verify whether NVIDIA Multi-Process Service (MPS) server is active to multiplex multiple inference processes onto shared CUDA contexts.",
        "k8s_equivalent": "kubectl get daemonset -n gpu-operator -l app=nvidia-mps-control-daemon",
        "dcgm_promql": "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
        "kubevram_cmd": "kubevram partition --gpu A10G-24GB --workload embeddings --tenants multi",
        "explanation": "MPS multiplexes multiple containers onto a single GPU while dividing physical device memory equally across configured replicas via CUDA_MPS_PINNED_DEVICE_MEM_LIMIT.",
    },
    {
        "id": "nvlink-status",
        "category": "Topology",
        "linux_cmd": "nvidia-smi nvlink -s -i 0",
        "linux_intent": "Query NVLink lane link states, error counters, and bidirectional status across all GPU interconnects.",
        "k8s_equivalent": "kubectl describe nodes | grep -iE 'nvlink|gpu.fabric'",
        "dcgm_promql": "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL > 0",
        "kubevram_cmd": "kubevram topology --fabric",
        "explanation": "Corrupted or down NVLink links degrade tensor parallelism all-reduce steps, severely impacting inference latency. DCGM exposes CRC flit error counters alerting on fabric degradation before silent data corruption occurs.",
    },
]


def translate_linux_intent(query: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Translates Linux / CUDA administration commands and mental models into
    Kubernetes + DCGM + KubeVRAM equivalents.
    """
    if not query or not query.strip():
        return list(BRIDGE_KNOWLEDGE_BASE)

    tokens = query.strip().lower().split()
    matched: List[Dict[str, str]] = []

    for entry in BRIDGE_KNOWLEDGE_BASE:
        combined_text = " ".join(str(val).lower() for val in entry.values())
        if all(token in combined_text for token in tokens):
            matched.append(entry)

    return matched
