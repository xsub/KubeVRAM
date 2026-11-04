"""
KubeVRAM OOM & Hardware Fault Correlation Engine.

Distinguishes between:
1. Kubernetes System RAM OOM (`OOMKilled`, Exit Code 137 — Linux cgroup host RAM limit)
2. CUDA VRAM OOM (`Error`, Exit Code 1 — GPU HBM exhaustion / vLLM KV-Cache over-allocation)
3. Time-Slicing Co-Tenant VRAM Collision (vLLM pre-allocation on shared physical VRAM)
4. NCCL Shared Memory (`/dev/shm`) Exhaustion (Exit Code 135 / SIGBUS on multi-GPU pods)
5. NVIDIA Hardware & Driver XID Faults (parsed XID code: silicon ECC/bus vs MMU/application faults)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from kubevram.models import ClusterNode, ContainerState, GPUDevice, InferencePod, RCAFinding

# Hardware/silicon/interconnect XIDs warranting node isolation or GPU reset/RMA evaluation
HARDWARE_ISOLATION_XIDS = {48, 63, 64, 74, 79, 92, 94, 95}

# Known XID human-readable descriptions
XID_DESCRIPTIONS: Dict[int, str] = {
    13: "Graphics Engine Exception (shader/kernel illegal instruction or out-of-bounds access)",
    31: "GPU Memory Page Fault / MMU Fault (application illegal memory address or driver mapping bug)",
    43: "GPU Stopped Processing (channel hang / watchdog timeout)",
    48: "Double-Bit ECC Error (uncorrectable HBM silicon error)",
    63: "ECC Page Retirement / Row Remapping Recording Event",
    64: "ECC Page Retirement / Row Remapping Failure",
    74: "NVLink Interconnect Error",
    79: "GPU Has Fallen Off the Bus (PCIe link / power / thermal hardware failure)",
    92: "High Single-Bit ECC Error Rate",
    94: "Contained ECC Error (uncorrectable memory error isolated to offending process)",
    95: "Uncontained ECC Error (uncorrectable memory error impacting shared GPU state)",
}

XID_LOG_PATTERNS = [
    re.compile(r"NVRM:\s*Xid\s*\([^)]*\)\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bXid\s*\([^)]*\)\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bXID[\s:=\-]+(\d+)\b", re.IGNORECASE),
]

CUDA_OOM_SIGNATURES = (
    "outofmemoryerror",
    "cuda out of memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "larger than the maximum number of tokens",
    "no available memory for the cache blocks",
    "out of memory on device",
)

SHM_SIGBUS_SIGNATURES = (
    "/dev/shm",
    "posix_fallocate failed",
    "killed by signal: bus error",
    "out of shared memory",
    "nccl warn call to posix_fallocate",
)

FRESHNESS_WINDOW_SECONDS = 300.0
RECOVERED_GRACE_SECONDS = 300.0


@dataclass
class ClusterLookupIndex:
    """Pre-indexed O(1) lookup tables for nodes and pod-to-GPU mappings."""

    nodes_by_name: Dict[str, ClusterNode] = field(default_factory=dict)
    gpus_by_node_idx: Dict[Tuple[str, str], GPUDevice] = field(default_factory=dict)
    gpus_by_pod_key: Dict[Tuple[str, str], List[GPUDevice]] = field(default_factory=dict)

    @classmethod
    def build(cls, nodes: List[ClusterNode]) -> "ClusterLookupIndex":
        nodes_by_name: Dict[str, ClusterNode] = {}
        gpus_by_node_idx: Dict[Tuple[str, str], GPUDevice] = {}
        gpus_by_pod_key: Dict[Tuple[str, str], List[GPUDevice]] = {}

        for node in nodes:
            nodes_by_name[node.name] = node
            for gpu in node.gpus:
                gpus_by_node_idx[(node.name, gpu.index)] = gpu
                for alloc in gpu.allocated_pods:
                    gpus_by_pod_key.setdefault((node.name, alloc), []).append(gpu)

        return cls(
            nodes_by_name=nodes_by_name,
            gpus_by_node_idx=gpus_by_node_idx,
            gpus_by_pod_key=gpus_by_pod_key,
        )


@lru_cache(maxsize=8192)
def _parse_iso_timestamp_str(ts: str) -> Optional[datetime]:
    cleaned = ts.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _parse_iso_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    return _parse_iso_timestamp_str(ts)


def _extract_log_xids(joined_logs: str, joined_logs_lower: Optional[str] = None) -> List[int]:
    lowered = joined_logs_lower if joined_logs_lower is not None else joined_logs.lower()
    if "xid" not in lowered:
        return []
    found: List[int] = []
    for pat in XID_LOG_PATTERNS:
        for match in pat.finditer(joined_logs):
            code = int(match.group(1))
            if code > 0 and code not in found:
                found.append(code)
    return found


def _find_node(
    nodes: List[ClusterNode],
    node_name: str,
    lookup_index: Optional[ClusterLookupIndex] = None,
) -> Optional[ClusterNode]:
    if lookup_index is not None:
        return lookup_index.nodes_by_name.get(node_name)
    for node in nodes:
        if node.name == node_name:
            return node
    return None


def _find_pod_gpus(
    node: Optional[ClusterNode],
    pod: InferencePod,
    lookup_index: Optional[ClusterLookupIndex] = None,
) -> List[GPUDevice]:
    if not node:
        return []
    if lookup_index is not None:
        seen_ids = set()
        matched: List[GPUDevice] = []
        for idx in pod.gpu_indices:
            g = lookup_index.gpus_by_node_idx.get((node.name, idx))
            if g is not None and id(g) not in seen_ids:
                seen_ids.add(id(g))
                matched.append(g)
        pod_ref = f"{pod.namespace}/{pod.name}"
        for g in lookup_index.gpus_by_pod_key.get((node.name, pod_ref), ()):
            if id(g) not in seen_ids:
                seen_ids.add(id(g))
                matched.append(g)
        return matched

    pod_ref = f"{pod.namespace}/{pod.name}"
    gpu_idx_set = set(pod.gpu_indices)
    return [
        gpu
        for gpu in node.gpus
        if gpu.index in gpu_idx_set or pod_ref in gpu.allocated_pods
    ]


def _resolve_primary_container(pod: InferencePod) -> Tuple[Optional[str], Optional[ContainerState]]:
    """Finds the failing or primary container state and its resolved container name."""
    chosen_state: Optional[ContainerState] = None
    for c in pod.containers:
        if c.exit_code != 0 or c.termination_reason not in ("None", "Completed", "") or c.waiting_reason:
            chosen_state = c
            break
    if chosen_state is None and pod.containers:
        chosen_state = pod.containers[0]

    resolved_name = pod.container_name or (chosen_state.name if chosen_state else None)
    return resolved_name, chosen_state


def _evaluate_freshness(
    terminated_at_str: Optional[str],
    measured_at_str: Optional[str],
) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    """
    Evaluates whether telemetry measurement aligns temporally with container termination.
    Returns (is_fresh, delta_seconds, gap_note).
    """
    term_dt = _parse_iso_timestamp(terminated_at_str)
    meas_dt = _parse_iso_timestamp(measured_at_str)
    if term_dt is None or meas_dt is None:
        return (
            None,
            None,
            "Missing termination or metric scrape timestamps; cannot verify metric freshness at crash time.",
        )
    delta = abs((meas_dt - term_dt).total_seconds())
    if delta <= FRESHNESS_WINDOW_SECONDS:
        return True, delta, None
    return (
        False,
        delta,
        f"Stale telemetry correlation: metric timestamp differs from termination by {int(delta)}s (> {int(FRESHNESS_WINDOW_SECONDS)}s).",
    )


def _calibrate_confidence(
    base_score: float,
    telemetry_fresh: Optional[bool],
    evidence_gaps: List[str],
) -> float:
    """Applies evidence-gap and metric-staleness caps so confidence is never overstated."""
    score = base_score
    if telemetry_fresh is True:
        score += 0.04
    elif telemetry_fresh is None:
        score = min(score, 0.84)
    else:  # stale telemetry
        score = min(score - 0.18, 0.70)

    if len(evidence_gaps) >= 2:
        score -= 0.05

    return round(max(0.35, min(0.94, score)), 2)


def _build_workload_memory_remediation(
    pod: InferencePod,
    container_name: Optional[str],
    req_gi: int,
    lim_gi: int,
    evidence_gaps: List[str],
) -> List[str]:
    """Generates owner- and container-aware Kubernetes remediation commands without guessing names."""
    patchable_kinds = {"deployment", "statefulset", "daemonset"}
    owner_kind_norm = (pod.owner_kind or "").strip().lower()
    if owner_kind_norm in patchable_kinds and pod.owner_name and container_name:
        return [
            f"kubectl patch {owner_kind_norm} -n {pod.namespace} {pod.owner_name} -p "
            f'\'{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"{container_name}",'
            f'"resources":{{"requests":{{"memory":"{req_gi}Gi"}},"limits":{{"memory":"{lim_gi}Gi"}}}}}}]}}}}}}}}\''
        ]
    if owner_kind_norm == "job" and pod.owner_name:
        cname_hint = f" (container '{container_name}')" if container_name else ""
        return [
            f"# Job pod templates are immutable; update memory limits{cname_hint} in Job '{pod.owner_name}' manifest and replace:",
            f"kubectl get job -n {pod.namespace} {pod.owner_name} -o yaml > job-{pod.owner_name}.yaml && kubectl replace --force -f job-{pod.owner_name}.yaml",
        ]

    evidence_gaps.append(
        "Missing workload ownerReference or container name metadata; remediation uses dynamic owner/container lookup."
    )
    return [
        f"kubectl get pod -n {pod.namespace} {pod.name} "
        "-o jsonpath='Owner={.metadata.ownerReferences[0].kind}/{.metadata.ownerReferences[0].name} Containers={.spec.containers[*].name}\\n'",
    ]


def _build_workload_env_remediation(
    pod: InferencePod,
    container_name: Optional[str],
    env_assignment: str,
    evidence_gaps: List[str],
) -> str:
    patchable_kinds = {"deployment", "statefulset", "daemonset"}
    owner_kind_norm = (pod.owner_kind or "").strip().lower()
    if owner_kind_norm in patchable_kinds and pod.owner_name:
        c_flag = f" --containers={container_name}" if container_name else ""
        return f"kubectl set env {owner_kind_norm}/{pod.owner_name} -n {pod.namespace}{c_flag} {env_assignment}"
    evidence_gaps.append(
        "Missing controller ownerReference metadata; cannot target a Deployment/StatefulSet directly."
    )
    return (
        f"kubectl get pod -n {pod.namespace} {pod.name} "
        "-o jsonpath='{.metadata.ownerReferences[0].kind}/{.metadata.ownerReferences[0].name}'"
    )


def correlate_pod_failure(
    pod: InferencePod,
    nodes: List[ClusterNode],
    lookup_index: Optional[ClusterLookupIndex] = None,
) -> Optional[RCAFinding]:
    """
    Correlates K8s Pod termination state, per-container status, cAdvisor host RAM,
    DCGM GPU metrics, timestamps, node sharing strategy, and container stderr logs.
    """
    node = _find_node(nodes, pod.node_name, lookup_index=lookup_index)
    gpus = _find_pod_gpus(node, pod, lookup_index=lookup_index)

    container_name, primary_container = _resolve_primary_container(pod)
    effective_exit_code = (
        primary_container.exit_code
        if (primary_container is not None and pod.last_exit_code == 0 and primary_container.exit_code != 0)
        else pod.last_exit_code
    )
    effective_reason = (
        primary_container.termination_reason
        if (
            primary_container is not None
            and pod.last_termination_reason in ("None", "")
            and primary_container.termination_reason not in ("None", "")
        )
        else pod.last_termination_reason
    )
    terminated_at = pod.last_terminated_at or (
        primary_container.terminated_at if primary_container is not None else None
    )
    measured_at = (
        pod.metrics_measured_at
        or next((g.measured_at for g in gpus if g.measured_at), None)
        or (node.measured_at if node else None)
    )

    gpu_xids: List[int] = []
    for g in gpus:
        if g.xid_error > 0 and g.xid_error not in gpu_xids:
            gpu_xids.append(g.xid_error)

    status_norm = (pod.status or "").strip()

    # 1. Active GPU XID check runs BEFORE skipping Running pods so silent hardware faults are caught
    has_active_gpu_xid = len(gpu_xids) > 0

    # 2. Filter out clean completions, normal pending pods, and recovered healthy pods
    waiting_err = primary_container.waiting_reason if primary_container else None

    if not has_active_gpu_xid:
        # Clean batch/job completion (Exit 0) is NOT an error
        if status_norm in ("Succeeded", "Completed") and effective_exit_code == 0 and effective_reason in ("Completed", "None", ""):
            return None

        # Pending pod with no container crash or config/image error
        if status_norm in ("Pending", "ContainerCreating", "PodInitializing"):
            if pod.restarts == 0 and effective_exit_code == 0 and not waiting_err:
                return None

        # Healthy Running pod
        if status_norm == "Running" and not waiting_err:
            all_containers_healthy = (
                all(c.state == "running" and c.ready for c in pod.containers)
                if pod.containers
                else True
            )
            if pod.restarts == 0 and effective_exit_code == 0 and all_containers_healthy:
                return None
            if effective_exit_code == 0 and effective_reason in ("Completed", "None", "") and all_containers_healthy:
                return None
            # If the pod restarted previously, check if it has since recovered cleanly
            if all_containers_healthy and terminated_at and measured_at:
                term_dt = _parse_iso_timestamp(terminated_at)
                meas_dt = _parse_iso_timestamp(measured_at)
                if term_dt and meas_dt and (meas_dt - term_dt).total_seconds() > RECOVERED_GRACE_SECONDS:
                    return None

    joined_logs = "\n".join(pod.recent_logs)
    joined_logs_lower = joined_logs.lower()
    log_xids = _extract_log_xids(joined_logs, joined_logs_lower)
    ecc_log_present = "uncorrectable ecc error" in joined_logs_lower
    has_crash_xid = (not has_active_gpu_xid) and (len(log_xids) > 0 or ecc_log_present)

    max_vram_pct = max((g.vram_util_pct for g in gpus), default=0.0)
    used_vram_mib = max((g.used_vram_mib for g in gpus), default=0)
    total_vram_mib = max((g.total_vram_mib for g in gpus), default=0)
    sharing_mode = node.sharing_strategy if node else "unknown"

    evidence_gaps: List[str] = []
    if node is None:
        evidence_gaps.append(f"Node '{pod.node_name}' metadata not found in cluster snapshot.")

    # For a Running pod with an active DCGM XID, freshness compares XID timestamp vs scrape timestamp
    xid_ts = next((g.xid_timestamp for g in gpus if g.xid_error > 0 and g.xid_timestamp), None)
    telemetry_fresh, delta_sec, freshness_gap = _evaluate_freshness(
        terminated_at or xid_ts, measured_at
    )
    if freshness_gap:
        evidence_gaps.append(freshness_gap)

    # =========================================================================
    # Case 1: NVIDIA XID Fault (DCGM active XID or parsed log XID / ECC error)
    # =========================================================================
    if has_active_gpu_xid or (status_norm != "Running" and has_crash_xid):
        # Prefer parsed log XID when it matches or specifies the pod's exact fault;
        # otherwise use DCGM GPU XID; never invent XID 95 when another XID or none is present.
        resolved_xid: Optional[int] = None
        if log_xids and gpu_xids:
            common = [x for x in log_xids if x in gpu_xids]
            resolved_xid = common[0] if common else gpu_xids[0]
        elif gpu_xids:
            resolved_xid = gpu_xids[0]
        elif log_xids:
            resolved_xid = log_xids[0]

        is_hw_xid = (resolved_xid in HARDWARE_ISOLATION_XIDS) if resolved_xid is not None else ecc_log_present
        xid_desc = (
            XID_DESCRIPTIONS.get(resolved_xid, "NVIDIA Driver / GPU XID Event")
            if resolved_xid is not None
            else "Uncorrectable ECC Hardware Error (XID code not reported in DCGM/logs)"
        )

        # Calibrate confidence based on corroboration between DCGM and container logs
        if gpu_xids and (log_xids or ecc_log_present):
            base_conf = 0.90
        elif gpu_xids and status_norm == "Running":
            base_conf = 0.80  # Active hardware fault on attached GPU, pod still running
        elif gpu_xids:
            base_conf = 0.84
        else:
            base_conf = 0.76  # Log-only XID without DCGM gauge confirmation
            evidence_gaps.append("DCGM_FI_DEV_XID_ERRORS is 0; XID derived from container log stream only.")

        conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)
        xid_label = f"XID {resolved_xid}" if resolved_xid is not None else "ECC Fault"

        tech_ev = [
            f"DCGM Metric: DCGM_FI_DEV_XID_ERRORS = {gpu_xids[0] if gpu_xids else 0} ({xid_desc})",
            f"Pod / Container State: status={pod.status}, exitCode={effective_exit_code} (Reason: {effective_reason})",
            f"Host RAM (cAdvisor): {pod.mem_used_gi:.1f} GiB / {pod.mem_limit_gi:.1f} GiB",
        ]
        if delta_sec is not None:
            tech_ev.append(f"Temporal Correlation: |metric_ts - event_ts| = {int(delta_sec)}s")
        if pod.recent_logs:
            tech_ev.append(f"Log Signature: '{pod.recent_logs[-1]}'")

        if is_hw_xid:
            severity = "HIGH" if resolved_xid == 94 else "CRITICAL"
            category = (
                f"NVIDIA_HARDWARE_XID_FAULT (XID {resolved_xid})"
                if resolved_xid is not None
                else "NVIDIA_HARDWARE_ECC_FAULT"
            )
            tag_val = f"XID-{resolved_xid}" if resolved_xid is not None else "ECC-FAULT"
            remediation_cmds = [
                f"kubectl cordon {pod.node_name}",
                f"kubectl annotate node {pod.node_name} nvidia.com/gpu-xid-fault='{tag_val}' --overwrite",
                f"kubectl delete pod -n {pod.namespace} {pod.name} --wait=false",
            ]
            yaml_patch = (
                "# Node Taint to Isolate Faulty GPU Silicon:\n"
                f"# kubectl taint nodes {pod.node_name} nvidia.com/gpu-xid-fault={tag_val}:NoSchedule"
            )
            root_cause = (
                f"GPU device {pod.gpu_indices} on node '{pod.node_name}' reported hardware fault {xid_label} "
                f"({xid_desc}). Workloads scheduled onto this GPU risk silent corruption or repeated crashes "
                "until the node is cordoned and GPU reset/RMA diagnostics complete."
            )
        else:
            severity = "HIGH"
            category = f"NVIDIA_GPU_MMU_OR_DRIVER_XID (XID {resolved_xid})"
            log_cmd = (
                f"kubectl logs -n {pod.namespace} {pod.name} -c {container_name} --previous"
                if container_name
                else f"kubectl logs -n {pod.namespace} {pod.name} --previous"
            )
            remediation_cmds = [
                log_cmd,
                f"kubectl logs -n gpu-operator -l app=nvidia-driver-daemonset --tail=100",
            ]
            yaml_patch = (
                "# Application/MMU XID fault (e.g., XID 31/13/43): do NOT RMA hardware before inspecting\n"
                "# custom CUDA/Triton kernels, --enforce-eager mode, and NVIDIA driver compatibility."
            )
            root_cause = (
                f"Container triggered {xid_label} ({xid_desc}) on node '{pod.node_name}'. Unlike XID 48/79/95, "
                f"{xid_label} typically indicates an application-level illegal GPU memory access or kernel bug "
                "rather than permanent HBM silicon failure."
            )

        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category=category,
            severity=severity,
            confidence=conf,
            summary=f"{xid_label} ({xid_desc}) affecting pod {pod.namespace}/{pod.name} on {pod.node_name}",
            technical_evidence=tech_ev,
            root_cause=root_cause,
            linux_equivalent_cmd="dmesg -T | grep -E 'NVRM: Xid|ECC' && nvidia-smi -q -d ECC,RETIRED_PAGES",
            k8s_remediation_cmds=remediation_cmds,
            yaml_patch=yaml_patch,
            evidence_gaps=evidence_gaps,
        )

    # =========================================================================
    # Case 2: Kubernetes System RAM OOMKilled (Exit Code 137 / reason=OOMKilled)
    # =========================================================================
    mem_ratio = (pod.mem_used_gi / pod.mem_limit_gi) if pod.mem_limit_gi > 0 else 0.0
    has_oom_log = (
        "memory cgroup out of memory" in joined_logs_lower
        or "oom-kill" in joined_logs_lower
        or joined_logs_lower.strip().endswith("killed")
    )
    is_confirmed_k8s_oom = (
        effective_reason == "OOMKilled"
        or status_norm == "OOMKilled"
        or (effective_exit_code == 137 and (mem_ratio >= 0.90 or has_oom_log))
    )

    if is_confirmed_k8s_oom:
        base_conf = 0.89 if (effective_reason == "OOMKilled" and mem_ratio >= 0.90) else 0.80
        req_gi = max(16, int(round(max(pod.mem_limit_gi, pod.mem_used_gi) * 1.5)))
        lim_gi = max(32, int(round(max(pod.mem_limit_gi, pod.mem_used_gi) * 2.0)))
        remediation_cmds = _build_workload_memory_remediation(
            pod, container_name, req_gi=req_gi, lim_gi=lim_gi, evidence_gaps=evidence_gaps
        )
        conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)

        tech_ev = [
            f"K8s Termination State: reason={effective_reason}, exitCode={effective_exit_code} (Linux cgroup OOM killer)",
            f"Host System RAM (cAdvisor): {pod.mem_used_gi:.1f} GiB / {pod.mem_limit_gi:.1f} GiB ({mem_ratio * 100:.1f}% of limit)",
            f"GPU Framebuffer (DCGM_FI_DEV_FB_USED): {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}%)",
        ]
        if delta_sec is not None:
            tech_ev.append(f"Temporal Correlation: |metric_ts - terminated_ts| = {int(delta_sec)}s")

        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="K8S_SYSTEM_RAM_OOMKILLED (Exit 137)",
            severity="HIGH",
            confidence=conf,
            summary=(
                f"Linux cgroup System RAM OOMKilled (Exit {effective_exit_code}): Host RAM hit "
                f"{pod.mem_used_gi:.1f}/{pod.mem_limit_gi:.1f} GiB while GPU VRAM was at {max_vram_pct:.1f}%."
            ),
            technical_evidence=tech_ev,
            root_cause=(
                "This is NOT a GPU VRAM OOM. The container exceeded its Kubernetes host RAM limit "
                f"(resources.limits.memory: {pod.mem_limit_gi:g}Gi) while staging/deserializing tensors in CPU RAM."
            ),
            linux_equivalent_cmd="dmesg -T | grep -i 'Memory cgroup out of memory' && free -h",
            k8s_remediation_cmds=remediation_cmds,
            yaml_patch=(
                "resources:\n"
                "  requests:\n"
                f"    memory: \"{req_gi}Gi\"\n"
                "  limits:\n"
                f"    memory: \"{lim_gi}Gi\""
            ),
            evidence_gaps=evidence_gaps,
        )

    # =========================================================================
    # Case 3: NCCL /dev/shm Bus Error (Exit Code 135 / SIGBUS or shm failure log)
    # =========================================================================
    has_shm_log = any(sig in joined_logs_lower for sig in SHM_SIGBUS_SIGNATURES)
    if effective_exit_code == 135 or effective_reason == "SIGBUS" or has_shm_log:
        corroborated = (effective_exit_code == 135 or effective_reason == "SIGBUS") and (
            has_shm_log or pod.shm_size_mib <= 64
        )
        base_conf = 0.88 if corroborated else 0.76
        conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)

        tech_ev = [
            f"K8s Container Exit Code: {effective_exit_code} (Reason: {effective_reason})",
            f"Configured /dev/shm Volume: {pod.shm_size_mib} MiB (Kubernetes default is 64 MiB)",
            f"Assigned GPUs: {len(pod.gpu_indices)} ({', '.join(pod.gpu_indices) or 'none'})",
        ]
        if pod.recent_logs:
            tech_ev.append(f"Log Signature: '{pod.recent_logs[-1]}'")

        cname_comment = f" (container '{container_name}')" if container_name else ""
        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category="NCCL_SHM_BUS_ERROR (Exit 135 / /dev/shm Exhausted)",
            severity="HIGH",
            confidence=conf,
            summary=(
                f"POSIX shared memory (/dev/shm) failure (Exit {effective_exit_code}): "
                f"/dev/shm configured at {pod.shm_size_mib} MiB"
            ),
            technical_evidence=tech_ev,
            root_cause=(
                "Multi-GPU NCCL collectives and PyTorch workers allocate shared memory segments in /dev/shm. "
                f"With /dev/shm limited to {pod.shm_size_mib} MiB, IPC allocation failed with SIGBUS."
            ),
            linux_equivalent_cmd="df -h /dev/shm && ipcs -m",
            k8s_remediation_cmds=[
                f"# Mount a memory-backed emptyDir (sizeLimit: 16Gi) at /dev/shm{cname_comment} for {pod.namespace}/{pod.name}"
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
            evidence_gaps=evidence_gaps,
        )

    # =========================================================================
    # Case 4 & 5: CUDA VRAM OOM (Requires CUDA/KV OOM evidence, NOT just >=94% VRAM!)
    # =========================================================================
    has_cuda_oom_log = any(sig in joined_logs_lower for sig in CUDA_OOM_SIGNATURES)
    # Never classify a pod whose logs show an unrelated error as CUDA OOM just because VRAM >= 94%
    # (vLLM routinely pre-allocates 90-95% VRAM during normal operation).

    if has_cuda_oom_log and status_norm != "Running":
        if sharing_mode == "time-slicing":
            co_tenants: List[str] = []
            pod_ref = f"{pod.namespace}/{pod.name}"
            for g in gpus:
                for alloc in g.allocated_pods:
                    if alloc != pod_ref and alloc not in co_tenants:
                        co_tenants.append(alloc)

            base_conf = 0.89 if (max_vram_pct >= 90.0 or co_tenants) else 0.78
            conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)

            tech_ev = [
                f"Node Sharing Strategy: nvidia.com/gpu.sharing-strategy = 'time-slicing' (replicas={node.replicas_per_gpu if node else 'unknown'})",
                f"DCGM Framebuffer: DCGM_FI_DEV_FB_USED = {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}%)",
                f"Pod Flag: --gpu-memory-utilization={pod.cli_args.get('--gpu-memory-utilization', 'default (0.90)')}",
                f"Co-located Tenant Pods on Same Physical GPU: {', '.join(co_tenants) if co_tenants else 'None observed at scrape time'}",
                f"Log Signature: '{pod.recent_logs[-1] if pod.recent_logs else 'CUDA out of memory'}'",
            ]
            if delta_sec is not None:
                tech_ev.append(f"Temporal Correlation: |metric_ts - terminated_ts| = {int(delta_sec)}s")

            return RCAFinding(
                pod_name=pod.name,
                namespace=pod.namespace,
                node_name=pod.node_name,
                failure_category=f"CUDA_VRAM_OOM_TIMESLICING_COLLISION (Exit {effective_exit_code})",
                severity="CRITICAL",
                confidence=conf,
                summary=(
                    f"Time-Slicing VRAM Collision on {pod.node_name}: Co-tenant "
                    f"({', '.join(co_tenants) or 'shared GPU workload'}) exhausted shared physical VRAM "
                    f"({used_vram_mib}/{total_vram_mib} MiB)."
                ),
                technical_evidence=tech_ev,
                root_cause=(
                    "NVIDIA Time-Slicing multiplexes compute contexts without isolating physical VRAM. "
                    "Co-located inference processes competed for the same framebuffer, causing CUDA memory "
                    "allocation to fail on startup."
                ),
                linux_equivalent_cmd="nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
                k8s_remediation_cmds=[
                    f"kubevram partition --gpu {node.gpu_model if node else 'L40S-48GB'} --workload vllm --tenants multi",
                    "# Isolate vLLM onto dedicated Full-GPU nodes or hardware-partitioned MIG slices (A100/H100).",
                ],
                yaml_patch=(
                    "affinity:\n"
                    "  nodeAffinity:\n"
                    "    requiredDuringSchedulingIgnoredDuringExecution:\n"
                    "      nodeSelectorTerms:\n"
                    "        - matchExpressions:\n"
                    "            - key: nvidia.com/gpu.sharing-strategy\n"
                    "              operator: NotIn\n"
                    "              values: [\"time-slicing\"]"
                ),
                evidence_gaps=evidence_gaps,
            )

        # Dedicated GPU / MIG CUDA VRAM OOM
        curr_len = pod.cli_args.get("--max-model-len", "unspecified")
        curr_util = pod.cli_args.get("--gpu-memory-utilization", "unspecified")
        base_conf = 0.88 if max_vram_pct >= 90.0 else 0.80
        env_cmd = _build_workload_env_remediation(
            pod, container_name, "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", evidence_gaps
        )
        conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)

        tech_ev = [
            f"K8s Termination State: reason={effective_reason}, exitCode={effective_exit_code} "
            f"(Host RAM at {pod.mem_used_gi:.1f}/{pod.mem_limit_gi:.1f} GiB)",
            f"DCGM Framebuffer: DCGM_FI_DEV_FB_USED = {used_vram_mib} MiB / {total_vram_mib} MiB ({max_vram_pct:.1f}%)",
            f"Engine Args: --max-model-len={curr_len}, --gpu-memory-utilization={curr_util}, "
            f"--kv-cache-dtype={pod.cli_args.get('--kv-cache-dtype', 'auto')}",
            f"Log Signature: '{pod.recent_logs[-1] if pod.recent_logs else 'CUDA out of memory'}'",
        ]
        if delta_sec is not None:
            tech_ev.append(f"Temporal Correlation: |metric_ts - terminated_ts| = {int(delta_sec)}s")

        return RCAFinding(
            pod_name=pod.name,
            namespace=pod.namespace,
            node_name=pod.node_name,
            failure_category=f"CUDA_VRAM_OOM_KV_CACHE_OVERFLOW (Exit {effective_exit_code})",
            severity="HIGH",
            confidence=conf,
            summary=(
                f"CUDA VRAM OOM (Exit {effective_exit_code}, NOT K8s OOMKilled): Model weights + KV cache "
                f"for --max-model-len={curr_len} exceeded available GPU HBM ({total_vram_mib} MiB total)."
            ),
            technical_evidence=tech_ev,
            root_cause=(
                f"GPU memory allocation failed inside the CUDA runtime while initializing weights or KV-cache "
                f"blocks (--max-model-len={curr_len}, --gpu-memory-utilization={curr_util}). Reduce context length, "
                "enable FP8 KV cache, or increase Tensor Parallelism."
            ),
            linux_equivalent_cmd="nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv",
            k8s_remediation_cmds=[
                "kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --dtype fp8 --kv-dtype fp8 --target-context 16384",
                env_cmd,
            ],
            yaml_patch=(
                "args:\n"
                "  - \"--gpu-memory-utilization=0.92\"\n"
                "  - \"--max-model-len=16384\"\n"
                "  - \"--kv-cache-dtype=fp8\""
            ),
            evidence_gaps=evidence_gaps,
        )

    # =========================================================================
    # Fallback: Abnormal Pod / Container Runtime Error (only when genuinely unhealthy)
    # =========================================================================
    if status_norm == "Running" and effective_exit_code == 0 and not waiting_err:
        return None

    if not pod.recent_logs:
        evidence_gaps.append("Container stderr logs unavailable; root cause cannot be narrowed from exit status alone.")

    base_conf = 0.68 if pod.recent_logs else 0.50
    conf = _calibrate_confidence(base_conf, telemetry_fresh, evidence_gaps)
    log_cmd = (
        f"kubectl logs -n {pod.namespace} {pod.name} -c {container_name} --previous"
        if container_name
        else f"kubectl logs -n {pod.namespace} {pod.name} --previous"
    )

    if effective_exit_code == 137 and effective_reason != "OOMKilled":
        category = "CONTAINER_SIGKILL (Exit 137)"
        root_cause = (
            "Container received SIGKILL (Exit 137) without Kubernetes OOMKilled termination reason or host RAM "
            "saturation; check livenessProbe timeouts, preemption, or manual process termination."
        )
    elif waiting_err in ("ImagePullBackOff", "ErrImagePull", "InvalidImageName", "CreateContainerConfigError"):
        category = f"POD_CONFIG_OR_IMAGE_ERROR ({waiting_err})"
        root_cause = f"Container failed before runtime startup ({waiting_err}); inspect Pod events and image pull secrets."
    elif effective_exit_code == 0:
        category = f"POD_ABNORMAL_STATE ({pod.status})"
        root_cause = f"Pod is in state '{pod.status}' without a non-zero container exit code; inspect Pod events."
    else:
        category = f"POD_RUNTIME_ERROR (Exit {effective_exit_code})"
        root_cause = (
            "Container exited with a non-OOM application or runtime error (note: high GPU VRAM allocation alone "
            "is normal for pre-allocating engines like vLLM and does not indicate CUDA OOM without an allocation failure)."
        )

    tech_ev = [
        f"Status: {pod.status}, Restarts: {pod.restarts}, ExitCode: {effective_exit_code}, Reason: {effective_reason}",
        f"GPU VRAM at Scrape: {used_vram_mib}/{total_vram_mib} MiB ({max_vram_pct:.1f}%)",
    ]
    if pod.recent_logs:
        tech_ev.append(f"Last Log Line: '{pod.recent_logs[-1]}'")

    return RCAFinding(
        pod_name=pod.name,
        namespace=pod.namespace,
        node_name=pod.node_name,
        failure_category=category,
        severity="MEDIUM",
        confidence=conf,
        summary=f"Pod {pod.namespace}/{pod.name} unhealthy ({category})",
        technical_evidence=tech_ev,
        root_cause=root_cause,
        linux_equivalent_cmd="journalctl -u containerd && nvidia-smi",
        k8s_remediation_cmds=[
            log_cmd,
            f"kubectl describe pod -n {pod.namespace} {pod.name}",
        ],
        yaml_patch="# Inspect container logs, probe thresholds, and application arguments",
        evidence_gaps=evidence_gaps,
    )


def triage_cluster(
    nodes: List[ClusterNode],
    pods: List[InferencePod],
    pod_filter: Optional[str] = None,
) -> List[RCAFinding]:
    """
    Runs the correlation engine across all unhealthy pods (or a specific filtered pod)
    using a pre-built O(1) node & GPU lookup index.
    """
    lookup_index = ClusterLookupIndex.build(nodes)
    findings: List[RCAFinding] = []
    for pod in pods:
        if pod_filter and pod_filter not in (pod.name, f"{pod.namespace}/{pod.name}"):
            continue
        finding = correlate_pod_failure(pod, nodes, lookup_index=lookup_index)
        if finding is not None:
            findings.append(finding)
    return findings
