"""
Security, RBAC, and Runtime Posture Auditor for Kubernetes GPU Workloads.

Inspects Kubernetes nodes and inference pods to detect privilege escalations,
explicit runtime mismatches, wildcard RBAC secret permissions across all ServiceAccounts,
lack of gang scheduling on multi-GPU workloads, IPC/SHM constraints, and NUMA affinity gaps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from kubevram.models import ClusterNode, InferencePod

_GPU_RESOURCE_COUNT_RE = re.compile(
    r"(?:nvidia\.com/[a-zA-Z0-9._\-]+|amd\.com/gpu|gpu)[ \t]*[:=][ \t]*(\d+)",
    re.IGNORECASE,
)
_FALLBACK_COLON_INT_RE = re.compile(r":[ \t]*(\d+)\b")


def parse_requested_gpu_count(requested_gpu_resource: str, gpu_indices: List[str]) -> int:
    """
    Parses the integer number of requested GPU / MIG devices from Pod resource strings
    (e.g., 'nvidia.com/gpu: 3', 'nvidia.com/gpu:5', 'nvidia.com/mig-2g.20gb: 1') and
    returns max(len(gpu_indices), parsed_count).
    """
    parsed_total = 0
    raw = (requested_gpu_resource or "").strip()
    if raw:
        matches = _GPU_RESOURCE_COUNT_RE.findall(raw)
        if matches:
            parsed_total = sum(int(m) for m in matches)
        elif raw.isdigit():
            parsed_total = int(raw)
        else:
            fallback = _FALLBACK_COLON_INT_RE.findall(raw)
            if fallback:
                parsed_total = sum(int(m) for m in fallback)

    return max(len(gpu_indices), parsed_total)


def audit_cluster_rbac(
    nodes: List[ClusterNode], pods: List[InferencePod]
) -> List[Dict[str, Any]]:
    """
    Inspects all nodes and pods and returns a list of security, RBAC, and runtime posture findings.
    """
    findings: List[Dict[str, Any]] = []
    nodes_by_name: Dict[str, ClusterNode] = {n.name: n for n in nodes}

    # 1. Audit Inference Pods
    for pod in pods:
        pod_resource = f"Pod/{pod.namespace}/{pod.name}"
        container_name = pod.container_name or (pod.containers[0].name if pod.containers else "inference")
        gpu_count = parse_requested_gpu_count(pod.requested_gpu_resource, pod.gpu_indices)
        node = nodes_by_name.get(pod.node_name)

        # GPU-SEC-001: Privileged container execution
        if pod.privileged:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-SEC-001",
                    "severity": "CRITICAL",
                    "issue": (
                        f"Pod '{pod.name}' runs with 'privileged: true'. Running privileged grants "
                        "host-level root capabilities and exposes ALL host GPU device nodes, bypassing "
                        "device-plugin / CDI isolation boundaries."
                    ),
                    "remediation": (
                        "Remove 'privileged: true' from container securityContext and inject GPUs via "
                        "NVIDIA Container Toolkit ('runtimeClassName: nvidia', nvidia-as-default runtime, or CDI)."
                    ),
                    "yaml_fix": (
                        "spec:\n"
                        "  runtimeClassName: nvidia\n"
                        "  containers:\n"
                        f"  - name: {container_name}\n"
                        "    securityContext:\n"
                        "      privileged: false\n"
                        "      allowPrivilegeEscalation: false\n"
                        "      capabilities:\n"
                        "        drop:\n"
                        "        - ALL"
                    ),
                }
            )

        # GPU-SEC-002: Explicit non-NVIDIA runtimeClassName or unverified missing runtimeClassName
        # Note: Clusters using NVIDIA as the default containerd runtime or CDI work without
        # Pod-level runtimeClassName: nvidia; do not false-positive on those configurations.
        if gpu_count > 0:
            rc = (pod.runtime_class_name or "").strip()
            rc_lower = rc.lower()
            node_supports_implicit_gpu = bool(
                node is not None
                and (
                    node.default_runtime_is_nvidia
                    or node.cdi_enabled
                    or (node.runtime_class or "").strip().lower() in ("nvidia-default", "cdi", "none-cdi")
                )
            )
            running_unprivileged_ok = (
                not rc
                and not pod.privileged
                and pod.status == "Running"
                and pod.restarts == 0
                and pod.last_exit_code == 0
            )

            if rc and rc_lower not in ("nvidia", "nvidia-cdi", "nvidia-experimental"):
                if not (node is not None and node.cdi_enabled):
                    findings.append(
                        {
                            "resource": pod_resource,
                            "namespace": pod.namespace,
                            "check_id": "GPU-SEC-002",
                            "severity": "HIGH",
                            "issue": (
                                f"Pod '{pod.name}' requests GPU resource '{pod.requested_gpu_resource}' "
                                f"with non-NVIDIA runtimeClassName='{rc}' on a node without CDI enabled."
                            ),
                            "remediation": (
                                "Set 'spec.runtimeClassName: nvidia' or enable CDI device injection on target GPU nodes."
                            ),
                            "yaml_fix": "spec:\n  runtimeClassName: nvidia",
                        }
                    )
            elif not rc and not node_supports_implicit_gpu and not running_unprivileged_ok:
                severity = "HIGH" if pod.privileged else "LOW"
                findings.append(
                    {
                        "resource": pod_resource,
                        "namespace": pod.namespace,
                        "check_id": "GPU-SEC-002",
                        "severity": severity,
                        "issue": (
                            f"Pod '{pod.name}' requests GPU resource '{pod.requested_gpu_resource}' without "
                            "declaring 'runtimeClassName: nvidia' on a node not marked as NVIDIA-default or CDI-enabled."
                        ),
                        "remediation": (
                            "Declare 'spec.runtimeClassName: nvidia' for portability, or verify that containerd "
                            "configures 'nvidia' as the default runtime or enables CDI on target nodes."
                        ),
                        "yaml_fix": "spec:\n  runtimeClassName: nvidia",
                    }
                )

        # GPU-RBAC-003: Wildcard secret access on ANY ServiceAccount (default or custom)
        if pod.wildcard_secrets:
            sa_name = pod.service_account or "default"
            sa_context = (
                "the namespace 'default' ServiceAccount"
                if sa_name == "default"
                else f"ServiceAccount '{sa_name}'"
            )
            recommended_sa = f"{pod.name}-restricted-sa" if sa_name != "default" else f"{pod.name}-sa"
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-RBAC-003",
                    "severity": "CRITICAL",
                    "issue": (
                        f"Pod '{pod.name}' runs under {sa_context} with wildcard ('*') secret access. "
                        "Remote code execution or SSRF in model-loading / tool-calling pipelines could "
                        "exfiltrate all namespace or cluster secrets."
                    ),
                    "remediation": (
                        f"Remove wildcard secret verbs/resourceNames from RoleBindings bound to '{sa_name}', "
                        "scope secret access via explicit 'resourceNames', and set 'automountServiceAccountToken: false' "
                        "if Kubernetes API access is unnecessary."
                    ),
                    "yaml_fix": (
                        "apiVersion: v1\n"
                        "kind: ServiceAccount\n"
                        "metadata:\n"
                        f"  name: {recommended_sa}\n"
                        f"  namespace: {pod.namespace}\n"
                        "automountServiceAccountToken: false\n"
                        "---\n"
                        "spec:\n"
                        f"  serviceAccountName: {recommended_sa}\n"
                        "  automountServiceAccountToken: false"
                    ),
                }
            )

        # GPU-SCHED-004: Multi-GPU pod (>= 2 GPUs) without Gang Scheduling
        is_multi_gpu = gpu_count >= 2
        if is_multi_gpu and not pod.pod_group:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-SCHED-004",
                    "severity": "MEDIUM",
                    "issue": (
                        f"Multi-GPU workload '{pod.name}' ({gpu_count} GPUs) has no gang-scheduling "
                        "'pod_group' configured. Under contention, partial GPU reservation across replicas "
                        "or multi-node workers can cause scheduling deadlocks."
                    ),
                    "remediation": (
                        "Configure Volcano, Kueue, or Coscheduling gang-scheduling annotations "
                        "('scheduling.k8s.io/group-name') for all-or-nothing placement."
                    ),
                    "yaml_fix": (
                        "metadata:\n"
                        "  annotations:\n"
                        f"    scheduling.k8s.io/group-name: {pod.name}-gang\n"
                        "  labels:\n"
                        f"    pod-group: {pod.name}-gang"
                    ),
                }
            )

        # GPU-IPC-006: Inadequate /dev/shm for multi-GPU NCCL communication
        if is_multi_gpu and pod.shm_size_mib <= 64:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-IPC-006",
                    "severity": "HIGH",
                    "issue": (
                        f"Multi-GPU pod '{pod.name}' ({gpu_count} GPUs) has only {pod.shm_size_mib} MiB /dev/shm "
                        "(Kubernetes default is 64 MiB). NCCL shared-memory transport allocates multi-gigabyte IPC "
                        "buffers and crashes with SIGBUS (Exit Code 135) when /dev/shm fills."
                    ),
                    "remediation": (
                        "Mount an emptyDir volume with medium: Memory at /dev/shm (sizeLimit >= 8Gi–16Gi)."
                    ),
                    "yaml_fix": (
                        "spec:\n"
                        "  volumes:\n"
                        "  - name: dshm\n"
                        "    emptyDir:\n"
                        "      medium: Memory\n"
                        "      sizeLimit: 16Gi\n"
                        "  containers:\n"
                        f"  - name: {container_name}\n"
                        "    volumeMounts:\n"
                        "    - mountPath: /dev/shm\n"
                        "      name: dshm"
                    ),
                }
            )

    # 2. Audit Cluster Nodes
    for node in nodes:
        node_resource = f"Node/{node.name}"

        # GPU-TOPO-005: Missing single-NUMA node policy
        if node.numa_policy == "none":
            findings.append(
                {
                    "resource": node_resource,
                    "namespace": "",
                    "check_id": "GPU-TOPO-005",
                    "severity": "MEDIUM",
                    "issue": (
                        f"GPU worker node '{node.name}' has numa_policy='{node.numa_policy}'. "
                        "Without 'single-numa-node', Kubelet Topology Manager does not align CPU core "
                        "affinity with PCIe GPU NUMA sockets."
                    ),
                    "remediation": (
                        "Configure Kubelet Topology Manager with 'topologyManagerPolicy: single-numa-node' "
                        "and 'cpuManagerPolicy: static' in KubeletConfiguration."
                    ),
                    "yaml_fix": (
                        "apiVersion: kubelet.config.k8s.io/v1beta1\n"
                        "kind: KubeletConfiguration\n"
                        "topologyManagerPolicy: single-numa-node\n"
                        "topologyManagerScope: container\n"
                        "cpuManagerPolicy: static"
                    ),
                }
            )

    return findings
