"""
Security, RBAC, and Runtime Posture Auditor for Kubernetes GPU Workloads.

Inspects Kubernetes nodes and inference pods to detect privilege escalations,
missing runtime classes, excessive RBAC permissions, lack of gang scheduling,
IPC/SHM constraints, and suboptimal NUMA affinity policies.
"""

from __future__ import annotations

from typing import Any, Dict, List

from kubevram.models import ClusterNode, InferencePod


def audit_cluster_rbac(
    nodes: List[ClusterNode], pods: List[InferencePod]
) -> List[Dict[str, Any]]:
    """
    Inspects all nodes and pods and returns a list of security, RBAC, and runtime posture findings.

    Each finding dictionary contains:
    - resource: e.g., 'Pod/ai-inference/vllm-mixtral-8x22b-tp2' or 'Node/gpu-node-l40s-ts-03'
    - namespace: str (pod namespace, or empty string for cluster-scoped resources like nodes)
    - check_id: e.g., 'GPU-SEC-001', 'GPU-SEC-002', 'GPU-RBAC-003', 'GPU-SCHED-004', 'GPU-TOPO-005'
    - severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
    - issue: clear description of the misconfiguration
    - remediation: actionable fix description
    - yaml_fix: short YAML snippet showing the declarative fix
    """
    findings: List[Dict[str, Any]] = []

    # 1. Audit Inference Pods
    for pod in pods:
        pod_resource = f"Pod/{pod.namespace}/{pod.name}"

        # GPU-SEC-001: Privileged container execution
        if pod.privileged:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-SEC-001",
                    "severity": "CRITICAL",
                    "issue": (
                        f"Pod '{pod.name}' runs with 'privileged: true' instead of using "
                        "'runtimeClassName: nvidia'. Running privileged grants raw host root "
                        "capabilities and bypasses container cgroup boundaries."
                    ),
                    "remediation": (
                        "Remove 'privileged: true' from container securityContext and configure "
                        "'runtimeClassName: nvidia' to enable secure, unprivileged GPU device "
                        "injection via NVIDIA Container Toolkit / CDI."
                    ),
                    "yaml_fix": (
                        "spec:\n"
                        "  runtimeClassName: nvidia\n"
                        "  containers:\n"
                        f"  - name: {pod.name}\n"
                        "    securityContext:\n"
                        "      privileged: false\n"
                        "      allowPrivilegeEscalation: false\n"
                        "      capabilities:\n"
                        "        drop:\n"
                        "        - ALL"
                    ),
                }
            )

        # GPU-SEC-002: Missing or invalid NVIDIA runtimeClassName
        if pod.runtime_class_name != "nvidia":
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-SEC-002",
                    "severity": "HIGH",
                    "issue": (
                        f"Pod '{pod.name}' requests GPU resource '{pod.requested_gpu_resource}' "
                        f"but specifies runtimeClassName='{pod.runtime_class_name}' instead of 'nvidia'. "
                        "Standard runc will fail to mount NVIDIA driver libraries and device nodes without privileged access."
                    ),
                    "remediation": (
                        "Set 'spec.runtimeClassName: nvidia' in the Pod manifest to invoke the "
                        "NVIDIA Container Runtime and CDI device injector."
                    ),
                    "yaml_fix": "spec:\n  runtimeClassName: nvidia",
                }
            )

        # GPU-RBAC-003: Default ServiceAccount with wildcard secret access
        if pod.service_account == "default" and pod.wildcard_secrets:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-RBAC-003",
                    "severity": "CRITICAL",
                    "issue": (
                        f"Pod '{pod.name}' runs under the default ServiceAccount while possessing "
                        "wildcard secret access permissions ('*'). Untrusted inference inputs or prompt "
                        "injection could lead to arbitrary code execution and full cluster secret exfiltration."
                    ),
                    "remediation": (
                        "Create a dedicated, restricted ServiceAccount for inference workloads, "
                        "set 'automountServiceAccountToken: false', and scope RBAC permissions to strictly "
                        "required secrets via RoleBinding."
                    ),
                    "yaml_fix": (
                        "apiVersion: v1\n"
                        "kind: ServiceAccount\n"
                        "metadata:\n"
                        f"  name: {pod.name}-sa\n"
                        f"  namespace: {pod.namespace}\n"
                        "automountServiceAccountToken: false\n"
                        "---\n"
                        "spec:\n"
                        f"  serviceAccountName: {pod.name}-sa"
                    ),
                }
            )

        # GPU-SCHED-004: Multi-GPU pod without Gang Scheduling
        is_multi_gpu = len(pod.gpu_indices) > 1 or any(
            f": {count}" in pod.requested_gpu_resource for count in ["2", "4", "8", "16"]
        )
        if is_multi_gpu and not pod.pod_group:
            findings.append(
                {
                    "resource": pod_resource,
                    "namespace": pod.namespace,
                    "check_id": "GPU-SCHED-004",
                    "severity": "MEDIUM",
                    "issue": (
                        f"Multi-GPU distributed inference workload '{pod.name}' ({len(pod.gpu_indices)} GPUs) "
                        "is scheduled without a 'pod_group' gang-scheduling declaration. In loaded clusters, "
                        "partial scheduling of multi-GPU ranks causes deadlocks and GPU resource starvation."
                    ),
                    "remediation": (
                        "Configure Volcano or Coscheduling Gang Scheduling annotations "
                        "('scheduling.k8s.io/group-name') or PodGroup CRD to ensure all-or-nothing scheduling."
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
                        f"Multi-GPU inference pod '{pod.name}' has only {pod.shm_size_mib} MiB /dev/shm "
                        "(Kubernetes default 64 MiB). NCCL tensor-parallel communication rings require "
                        "gigabytes of shared memory for P2P buffer staging, causing SIGBUS (Exit Code 135)."
                    ),
                    "remediation": (
                        "Mount an emptyDir volume with medium: Memory to /dev/shm with a size limit "
                        "of at least 16Gi."
                    ),
                    "yaml_fix": (
                        "spec:\n"
                        "  volumes:\n"
                        "  - name: dshm\n"
                        "    emptyDir:\n"
                        "      medium: Memory\n"
                        "      sizeLimit: 16Gi\n"
                        "  containers:\n"
                        f"  - name: {pod.name}\n"
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
                        "Without 'single-numa-node', Kubelet Topology Manager fails to align CPU core "
                        "affinity with PCIe GPU sockets, leading to cross-socket UPI/QPI interconnect bottlenecks."
                    ),
                    "remediation": (
                        "Configure Kubelet Topology Manager with 'topologyManagerPolicy: single-numa-node' "
                        "and 'cpuManagerPolicy: static' in /var/lib/kubelet/config.yaml on the GPU worker node."
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
