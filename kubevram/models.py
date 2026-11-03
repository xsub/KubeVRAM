"""
Core domain models for KubeVRAM: GPU topology, DCGM telemetry, Inference Pods, and RCA findings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GPUDevice:
    """Represents a physical GPU or a hardware-partitioned MIG slice on a Kubernetes node."""

    index: str  # e.g., "0", "1", or "MIG-1:0 (3g.40gb)"
    uuid: str
    model: str
    total_vram_mib: int
    used_vram_mib: int
    gpu_util_pct: float
    sm_active_pct: float
    dram_active_pct: float
    temp_c: int
    power_w: int
    xid_error: int = 0
    mig_profile: Optional[str] = None
    numa_node: int = 0
    interconnect: str = "NVSwitch (900 GB/s)"
    allocated_pods: List[str] = field(default_factory=list)

    @property
    def free_vram_mib(self) -> int:
        return max(0, self.total_vram_mib - self.used_vram_mib)

    @property
    def vram_util_pct(self) -> float:
        if self.total_vram_mib <= 0:
            return 0.0
        return round((self.used_vram_mib / self.total_vram_mib) * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["free_vram_mib"] = self.free_vram_mib
        data["vram_util_pct"] = self.vram_util_pct
        return data


@dataclass
class ClusterNode:
    """Represents a Kubernetes GPU Worker Node and its NVIDIA GPU Operator stack state."""

    name: str
    gpu_model: str
    physical_gpus: int
    advertised_gpu_resources: Dict[str, int]
    sharing_strategy: str  # "none" | "mig" | "time-slicing" | "mps"
    mig_strategy: str  # "none" | "single" | "mixed"
    replicas_per_gpu: int
    interconnect_fabric: str  # e.g., "NVSwitch Gen4 (900 GB/s)", "NVLink Bridge", "PCIe Gen4 x16"
    numa_policy: str  # e.g., "single-numa-node", "best-effort", "none"
    driver_version: str
    cuda_version: str
    runtime_class: str
    schedulable: bool = True
    gpus: List[GPUDevice] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [g.to_dict() for g in self.gpus]
        return data


@dataclass
class InferencePod:
    """Represents an AI Inference Pod scheduled on Kubernetes alongside its container/cgroup telemetry."""

    name: str
    namespace: str
    node_name: str
    workload_engine: str  # "vllm" | "triton" | "kserve" | "ray-worker" | "pytorch-loader"
    model_name: str
    status: str  # "Running" | "CrashLoopBackOff" | "Error" | "Pending" | "OOMKilled"
    restarts: int
    last_exit_code: int
    last_termination_reason: str  # "Completed" | "OOMKilled" | "Error" | "SIGBUS" | "None"
    requested_gpu_resource: str  # e.g., "nvidia.com/gpu: 1" or "nvidia.com/mig-3g.40gb: 1"
    mem_request_gi: float
    mem_limit_gi: float
    mem_used_gi: float  # Host cgroup RAM usage (cAdvisor)
    shm_size_mib: int  # Size of /dev/shm mount in MiB (default 64Mi in K8s)
    gpu_indices: List[str]
    cli_args: Dict[str, str]
    recent_logs: List[str]
    service_account: str = "default"
    privileged: bool = False
    runtime_class_name: Optional[str] = "nvidia"
    pod_group: Optional[str] = None
    wildcard_secrets: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RCAFinding:
    """Structured Root Cause Analysis finding produced by the KubeVRAM OOM & Hardware Correlator."""

    pod_name: str
    namespace: str
    node_name: str
    failure_category: str
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    confidence: float
    summary: str
    technical_evidence: List[str]
    root_cause: str
    linux_equivalent_cmd: str
    k8s_remediation_cmds: List[str]
    yaml_patch: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
