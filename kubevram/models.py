"""
Core domain models for KubeVRAM: GPU topology, DCGM telemetry, Inference Pods, and RCA findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContainerState:
    """Per-container execution and last-termination state for reliable incident correlation."""

    name: str
    state: str = "running"  # "running" | "waiting" | "terminated"
    ready: bool = True
    restarts: int = 0
    exit_code: int = 0
    termination_reason: str = "None"  # "Completed" | "OOMKilled" | "Error" | "SIGBUS" | "None"
    waiting_reason: Optional[str] = None  # e.g., "CrashLoopBackOff", "ImagePullBackOff"
    terminated_at: Optional[str] = None  # ISO-8601 UTC timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "ready": self.ready,
            "restarts": self.restarts,
            "exit_code": self.exit_code,
            "termination_reason": self.termination_reason,
            "waiting_reason": self.waiting_reason,
            "terminated_at": self.terminated_at,
        }


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
    measured_at: Optional[str] = None  # ISO-8601 UTC timestamp of DCGM scrape
    xid_timestamp: Optional[str] = None  # ISO-8601 UTC timestamp of last XID event

    @property
    def free_vram_mib(self) -> int:
        return max(0, self.total_vram_mib - self.used_vram_mib)

    @property
    def vram_util_pct(self) -> float:
        if self.total_vram_mib <= 0:
            return 0.0
        return round((self.used_vram_mib / self.total_vram_mib) * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        total = self.total_vram_mib
        used = self.used_vram_mib
        free = total - used if total > used else 0
        util_pct = round((used / total) * 100.0, 1) if total > 0 else 0.0
        return {
            "index": self.index,
            "uuid": self.uuid,
            "model": self.model,
            "total_vram_mib": total,
            "used_vram_mib": used,
            "gpu_util_pct": self.gpu_util_pct,
            "sm_active_pct": self.sm_active_pct,
            "dram_active_pct": self.dram_active_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "xid_error": self.xid_error,
            "mig_profile": self.mig_profile,
            "numa_node": self.numa_node,
            "interconnect": self.interconnect,
            "allocated_pods": list(self.allocated_pods),
            "measured_at": self.measured_at,
            "xid_timestamp": self.xid_timestamp,
            "free_vram_mib": free,
            "vram_util_pct": util_pct,
        }


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
    default_runtime_is_nvidia: bool = False
    cdi_enabled: bool = False
    measured_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gpu_model": self.gpu_model,
            "physical_gpus": self.physical_gpus,
            "advertised_gpu_resources": dict(self.advertised_gpu_resources),
            "sharing_strategy": self.sharing_strategy,
            "mig_strategy": self.mig_strategy,
            "replicas_per_gpu": self.replicas_per_gpu,
            "interconnect_fabric": self.interconnect_fabric,
            "numa_policy": self.numa_policy,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "runtime_class": self.runtime_class,
            "schedulable": self.schedulable,
            "default_runtime_is_nvidia": self.default_runtime_is_nvidia,
            "cdi_enabled": self.cdi_enabled,
            "measured_at": self.measured_at,
            "gpus": [g.to_dict() for g in self.gpus],
        }


@dataclass
class InferencePod:
    """Represents an AI Inference Pod scheduled on Kubernetes alongside its container/cgroup telemetry."""

    name: str
    namespace: str
    node_name: str
    workload_engine: str  # "vllm" | "triton" | "kserve" | "ray-worker" | "pytorch-loader"
    model_name: str
    status: str  # "Running" | "CrashLoopBackOff" | "Error" | "Pending" | "OOMKilled" | "Succeeded" | "Failed"
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
    owner_kind: Optional[str] = None  # e.g., "Deployment", "StatefulSet", "Job", "DaemonSet"
    owner_name: Optional[str] = None  # Controller resource name
    container_name: Optional[str] = None  # Target inference container name
    containers: List[ContainerState] = field(default_factory=list)
    last_terminated_at: Optional[str] = None  # ISO-8601 UTC timestamp
    metrics_measured_at: Optional[str] = None  # ISO-8601 UTC timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "node_name": self.node_name,
            "workload_engine": self.workload_engine,
            "model_name": self.model_name,
            "status": self.status,
            "restarts": self.restarts,
            "last_exit_code": self.last_exit_code,
            "last_termination_reason": self.last_termination_reason,
            "requested_gpu_resource": self.requested_gpu_resource,
            "mem_request_gi": self.mem_request_gi,
            "mem_limit_gi": self.mem_limit_gi,
            "mem_used_gi": self.mem_used_gi,
            "shm_size_mib": self.shm_size_mib,
            "gpu_indices": list(self.gpu_indices),
            "cli_args": dict(self.cli_args),
            "recent_logs": list(self.recent_logs),
            "service_account": self.service_account,
            "privileged": self.privileged,
            "runtime_class_name": self.runtime_class_name,
            "pod_group": self.pod_group,
            "wildcard_secrets": self.wildcard_secrets,
            "owner_kind": self.owner_kind,
            "owner_name": self.owner_name,
            "container_name": self.container_name,
            "containers": [c.to_dict() for c in self.containers],
            "last_terminated_at": self.last_terminated_at,
            "metrics_measured_at": self.metrics_measured_at,
        }


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
    evidence_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pod_name": self.pod_name,
            "namespace": self.namespace,
            "node_name": self.node_name,
            "failure_category": self.failure_category,
            "severity": self.severity,
            "confidence": self.confidence,
            "summary": self.summary,
            "technical_evidence": list(self.technical_evidence),
            "root_cause": self.root_cause,
            "linux_equivalent_cmd": self.linux_equivalent_cmd,
            "k8s_remediation_cmds": list(self.k8s_remediation_cmds),
            "yaml_patch": self.yaml_patch,
            "evidence_gaps": list(self.evidence_gaps),
        }
