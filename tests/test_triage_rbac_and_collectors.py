"""
Semantic unit tests for OOM/XID triage correlation, RBAC security auditing,
live/snapshot collectors, and O(N + M) scaling performance.
"""

import contextlib
import io
import json
import tempfile
import time
import unittest

from kubevram.cli import group_pod_groups_by_node, main
from kubevram.collectors import (
    CollectorError,
    load_cluster_from_file,
    parse_dcgm_text_metrics,
    parse_k8s_objects_to_state,
)
from kubevram.models import ClusterNode, ContainerState, GPUDevice, InferencePod
from kubevram.oom_correlator import correlate_pod_failure, triage_cluster
from kubevram.rbac_auditor import audit_cluster_rbac, parse_requested_gpu_count


def _make_h100_node(
    name: str = "node-h100-01",
    used_mib: int = 78000,
    xid_error: int = 0,
    sharing: str = "none",
    default_rt_nvidia: bool = False,
    cdi_enabled: bool = False,
    measured_at: str = "2026-11-04T01:20:00Z",
) -> ClusterNode:
    return ClusterNode(
        name=name,
        gpu_model="NVIDIA-H100-80GB-HBM3",
        physical_gpus=1,
        advertised_gpu_resources={"nvidia.com/gpu": 1},
        sharing_strategy=sharing,
        mig_strategy="none",
        replicas_per_gpu=1,
        interconnect_fabric="NVSwitch",
        numa_policy="single-numa-node",
        driver_version="550.90.07",
        cuda_version="12.4",
        runtime_class="nvidia",
        default_runtime_is_nvidia=default_rt_nvidia,
        cdi_enabled=cdi_enabled,
        measured_at=measured_at,
        gpus=[
            GPUDevice(
                index="0",
                uuid=f"GPU-{name}-0",
                model="NVIDIA H100 80GB HBM3",
                total_vram_mib=81920,
                used_vram_mib=used_mib,
                gpu_util_pct=85.0,
                sm_active_pct=70.0,
                dram_active_pct=75.0,
                temp_c=62,
                power_w=550,
                xid_error=xid_error,
                allocated_pods=["ai/test-pod"],
                measured_at=measured_at,
                xid_timestamp=measured_at if xid_error > 0 else None,
            )
        ],
    )


def _make_pod(**overrides) -> InferencePod:
    base = dict(
        name="test-pod",
        namespace="ai",
        node_name="node-h100-01",
        workload_engine="vllm",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        status="Running",
        restarts=0,
        last_exit_code=0,
        last_termination_reason="None",
        requested_gpu_resource="nvidia.com/gpu: 1",
        mem_request_gi=32.0,
        mem_limit_gi=64.0,
        mem_used_gi=20.0,
        shm_size_mib=16384,
        gpu_indices=["0"],
        cli_args={"--gpu-memory-utilization": "0.95"},
        recent_logs=["INFO: Serving ready"],
        service_account="inference-sa",
        privileged=False,
        runtime_class_name="nvidia",
    )
    base.update(overrides)
    return InferencePod(**base)


class TestTriageCorrelator(unittest.TestCase):
    def test_succeeded_completed_pod_with_exit_0_is_not_flagged(self):
        node = _make_h100_node(used_mib=1024)
        pod = _make_pod(
            status="Succeeded",
            restarts=0,
            last_exit_code=0,
            last_termination_reason="Completed",
            recent_logs=["Batch inference completed cleanly"],
        )
        self.assertIsNone(correlate_pod_failure(pod, [node]))

    def test_unrelated_app_error_at_high_vram_is_not_classified_as_cuda_oom(self):
        # 78,643 MiB / 81,920 MiB = 96.0% VRAM (> 94%), normal for vLLM pre-allocation
        node = _make_h100_node(used_mib=78643)
        pod = _make_pod(
            status="CrashLoopBackOff",
            restarts=3,
            last_exit_code=1,
            last_termination_reason="Error",
            recent_logs=[
                "Traceback (most recent call last):",
                "huggingface_hub.utils.HfHubHTTPError: 401 Client Error: Unauthorized for url",
            ],
            last_terminated_at="2026-11-04T01:19:50Z",
            metrics_measured_at="2026-11-04T01:20:00Z",
        )
        finding = correlate_pod_failure(pod, [node])
        self.assertIsNotNone(finding)
        self.assertNotIn("CUDA_VRAM_OOM", finding.failure_category)
        self.assertIn("POD_RUNTIME_ERROR", finding.failure_category)

    def test_unrelated_error_on_multi_gpu_small_shm_is_not_classified_as_sigbus(self):
        node = _make_h100_node(used_mib=40000)
        pod = _make_pod(
            status="Error",
            restarts=1,
            last_exit_code=1,
            last_termination_reason="Error",
            shm_size_mib=64,
            gpu_indices=["0", "1"],
            requested_gpu_resource="nvidia.com/gpu: 2",
            recent_logs=["ValueError: Invalid --rope-scaling configuration"],
        )
        finding = correlate_pod_failure(pod, [node])
        self.assertIsNotNone(finding)
        self.assertNotIn("NCCL_SHM_BUS_ERROR", finding.failure_category)

    def test_log_with_xid_31_reports_xid_31_mmu_fault_not_xid_95(self):
        node = _make_h100_node(xid_error=0)
        pod = _make_pod(
            status="CrashLoopBackOff",
            restarts=2,
            last_exit_code=1,
            last_termination_reason="Error",
            recent_logs=[
                "NVRM: Xid (PCI:0000:81:00): 31, pid=4412, Ch 00000010, intr 00000000. MMU Fault: ENGINE GRAPHICS"
            ],
            last_terminated_at="2026-11-04T01:19:55Z",
            metrics_measured_at="2026-11-04T01:20:00Z",
        )
        finding = correlate_pod_failure(pod, [node])
        self.assertIsNotNone(finding)
        self.assertIn("XID 31", finding.failure_category)
        self.assertNotIn("95", finding.failure_category)
        self.assertEqual(finding.severity, "HIGH")
        # XID 31 is an application/MMU fault, so it should NOT cordon the node
        self.assertFalse(any("kubectl cordon" in cmd for cmd in finding.k8s_remediation_cmds))

    def test_running_pod_on_gpu_reporting_xid_95_is_not_skipped(self):
        node = _make_h100_node(xid_error=95)
        pod = _make_pod(
            status="Running",
            restarts=0,
            last_exit_code=0,
            last_termination_reason="None",
            recent_logs=["INFO: Serving requests"],
        )
        finding = correlate_pod_failure(pod, [node])
        self.assertIsNotNone(finding)
        self.assertIn("XID 95", finding.failure_category)
        self.assertEqual(finding.severity, "CRITICAL")

    def test_recovered_running_pod_with_old_restart_is_skipped(self):
        node = _make_h100_node(used_mib=75000, measured_at="2026-11-04T05:00:00Z")
        pod = _make_pod(
            status="Running",
            restarts=1,
            last_exit_code=1,
            last_termination_reason="Error",
            last_terminated_at="2026-11-04T01:00:00Z",
            metrics_measured_at="2026-11-04T05:00:00Z",
            containers=[ContainerState(name="vllm", state="running", ready=True, restarts=1, exit_code=1)],
        )
        self.assertIsNone(correlate_pod_failure(pod, [node]))

    def test_stale_timestamps_penalize_confidence_and_owner_metadata_targets_real_controller(self):
        node = _make_h100_node(used_mib=10000, measured_at="2026-11-04T04:00:00Z")
        pod_with_owner = _make_pod(
            status="OOMKilled",
            restarts=2,
            last_exit_code=137,
            last_termination_reason="OOMKilled",
            mem_limit_gi=16.0,
            mem_used_gi=16.0,
            owner_kind="StatefulSet",
            owner_name="qwen-serving-sts",
            container_name="qwen-worker",
            last_terminated_at="2026-11-04T01:00:00Z",  # 3 hours stale!
            metrics_measured_at="2026-11-04T04:00:00Z",
        )
        finding = correlate_pod_failure(pod_with_owner, [node])
        self.assertIsNotNone(finding)
        self.assertLessEqual(finding.confidence, 0.70)
        self.assertTrue(any("Stale telemetry" in g for g in finding.evidence_gaps))
        self.assertIn("kubectl patch statefulset -n ai qwen-serving-sts", finding.k8s_remediation_cmds[0])
        self.assertIn('"name":"qwen-worker"', finding.k8s_remediation_cmds[0])

        # Without owner/container metadata: never invent deployment/test-pod or container 'main'
        pod_no_owner = _make_pod(
            status="OOMKilled",
            restarts=1,
            last_exit_code=137,
            last_termination_reason="OOMKilled",
            mem_limit_gi=16.0,
            mem_used_gi=16.0,
            owner_kind=None,
            owner_name=None,
            container_name=None,
        )
        finding_no_owner = correlate_pod_failure(pod_no_owner, [node])
        self.assertIsNotNone(finding_no_owner)
        for cmd in finding_no_owner.k8s_remediation_cmds:
            self.assertNotIn('"name":"main"', cmd)
            self.assertNotIn("patch deployment -n ai test-pod", cmd)


class TestRBACAuditor(unittest.TestCase):
    def test_wildcard_secrets_detected_for_non_default_service_account(self):
        node = _make_h100_node()
        pod = _make_pod(service_account="custom-vllm-sa", wildcard_secrets=True, pod_group="pg-1")
        findings = audit_cluster_rbac([node], [pod])
        check_ids = [f["check_id"] for f in findings]
        self.assertIn("GPU-RBAC-003", check_ids)

    def test_missing_runtime_class_not_flagged_on_cdi_or_nvidia_default_or_healthy_running_pod(self):
        cdi_node = _make_h100_node(name="cdi-node", cdi_enabled=True)
        pod_cdi = _make_pod(node_name="cdi-node", runtime_class_name=None, status="Pending")
        self.assertEqual(audit_cluster_rbac([cdi_node], [pod_cdi]), [])

        def_rt_node = _make_h100_node(name="def-rt-node", default_rt_nvidia=True)
        pod_def_rt = _make_pod(node_name="def-rt-node", runtime_class_name=None, status="Pending")
        self.assertEqual(audit_cluster_rbac([def_rt_node], [pod_def_rt]), [])

        std_node = _make_h100_node(name="std-node")
        pod_running_ok = _make_pod(node_name="std-node", runtime_class_name=None, status="Running", privileged=False)
        self.assertEqual(audit_cluster_rbac([std_node], [pod_running_ok]), [])

        # Explicit wrong runtimeClassName IS flagged
        pod_runc = _make_pod(node_name="std-node", runtime_class_name="runc", status="Pending")
        runc_findings = audit_cluster_rbac([std_node], [pod_runc])
        self.assertIn("GPU-SEC-002", [f["check_id"] for f in runc_findings])

    def test_gpu_count_parsing_catches_3_and_5_gpus_and_ignores_mig_profile_digits(self):
        self.assertEqual(parse_requested_gpu_count("nvidia.com/gpu: 3", []), 3)
        self.assertEqual(parse_requested_gpu_count("nvidia.com/gpu:5", []), 5)
        self.assertEqual(parse_requested_gpu_count("nvidia.com/mig-2g.20gb: 1", []), 1)

        node = _make_h100_node()
        for count in (3, 5):
            pending_pod = _make_pod(
                name=f"pending-{count}gpu",
                status="Pending",
                gpu_indices=[],
                requested_gpu_resource=f"nvidia.com/gpu: {count}",
                shm_size_mib=64,
                pod_group=None,
            )
            findings = audit_cluster_rbac([node], [pending_pod])
            ids = {f["check_id"] for f in findings}
            self.assertIn("GPU-SCHED-004", ids)
            self.assertIn("GPU-IPC-006", ids)
            sched_issue = next(f["issue"] for f in findings if f["check_id"] == "GPU-SCHED-004")
            self.assertIn(f"({count} GPUs)", sched_issue)


class TestCollectorsAndSnapshots(unittest.TestCase):
    def test_parse_k8s_bundle_unwraps_replicaset_rbac_and_dcgm_metrics(self):
        node_items = [
            {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": "gpu-worker-1",
                    "labels": {
                        "nvidia.com/gpu.present": "true",
                        "nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
                        "nvidia.com/gpu.count": "2",
                        "nvidia.com/gpu.memory": "81920",
                        "nvidia.com/gpu.cdi": "true",
                    },
                },
                "spec": {},
                "status": {"allocatable": {"nvidia.com/gpu": "2"}},
            }
        ]
        rs_items = [
            {
                "apiVersion": "apps/v1",
                "kind": "ReplicaSet",
                "metadata": {
                    "name": "vllm-deploy-7f89b",
                    "namespace": "prod",
                    "ownerReferences": [{"kind": "Deployment", "name": "vllm-deploy"}],
                },
            }
        ]
        rbac_items = [
            {
                "kind": "Role",
                "metadata": {"name": "secret-reader-all", "namespace": "prod"},
                "rules": [{"apiGroups": [""], "resources": ["secrets"], "verbs": ["get", "list"]}],
            },
            {
                "kind": "RoleBinding",
                "metadata": {"name": "bind-sa", "namespace": "prod"},
                "roleRef": {"kind": "Role", "name": "secret-reader-all"},
                "subjects": [{"kind": "ServiceAccount", "name": "vllm-sa", "namespace": "prod"}],
            },
        ]
        pod_items = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "vllm-deploy-7f89b-abcde",
                    "namespace": "prod",
                    "ownerReferences": [{"kind": "ReplicaSet", "name": "vllm-deploy-7f89b"}],
                },
                "spec": {
                    "nodeName": "gpu-worker-1",
                    "serviceAccountName": "vllm-sa",
                    "containers": [
                        {
                            "name": "vllm-server",
                            "image": "vllm/vllm-openai:v0.6.4",
                            "args": ["--model=Qwen/Qwen2.5-32B-Instruct", "--max-model-len=32768"],
                            "resources": {
                                "requests": {"memory": "32Gi", "nvidia.com/gpu": "2"},
                                "limits": {"memory": "64Gi", "nvidia.com/gpu": "2"},
                            },
                            "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
                        }
                    ],
                    "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
                },
                "status": {
                    "phase": "Running",
                    "containerStatuses": [
                        {
                            "name": "vllm-server",
                            "ready": False,
                            "restartCount": 2,
                            "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                            "lastState": {
                                "terminated": {
                                    "exitCode": 1,
                                    "reason": "Error",
                                    "finishedAt": "2026-11-04T01:19:50Z",
                                }
                            },
                        }
                    ],
                },
            }
        ]
        dcgm_text = """
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-111",Hostname="gpu-worker-1",modelName="NVIDIA H100 80GB HBM3",exported_namespace="prod",exported_pod="vllm-deploy-7f89b-abcde"} 80500 1793755200000
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-111",Hostname="gpu-worker-1",modelName="NVIDIA H100 80GB HBM3",exported_namespace="prod",exported_pod="vllm-deploy-7f89b-abcde"} 1420 1793755200000
"""
        samples = parse_dcgm_text_metrics(dcgm_text)
        nodes, pods = parse_k8s_objects_to_state(
            node_items=node_items,
            pod_items=pod_items,
            rs_items=rs_items,
            rbac_items=rbac_items,
            dcgm_samples=samples,
            pod_logs_by_key={
                "prod/vllm-deploy-7f89b-abcde": [
                    "torch.cuda.OutOfMemoryError: CUDA out of memory."
                ]
            },
            scrape_timestamp="2026-11-04T01:20:00Z",
        )
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes[0].cdi_enabled)
        self.assertEqual(len(pods), 1)
        pod = pods[0]
        self.assertEqual(pod.owner_kind, "Deployment")
        self.assertEqual(pod.owner_name, "vllm-deploy")
        self.assertEqual(pod.container_name, "vllm-server")
        self.assertTrue(pod.wildcard_secrets)
        self.assertEqual(pod.shm_size_mib, 16384)

        # Verify --from-file round-trip on CLI
        bundle = {
            "nodes": [n.to_dict() for n in nodes],
            "pods": [p.to_dict() for p in pods],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as tmp:
            json.dump(bundle, tmp)
            tmp.flush()
            loaded_nodes, loaded_pods = load_cluster_from_file(tmp.name)
            self.assertEqual(len(loaded_nodes), 1)
            self.assertEqual(len(loaded_pods), 1)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["triage", "--from-file", tmp.name, "--format", "json"])
            self.assertEqual(code, 0)
            parsed = json.loads(buf.getvalue())
            self.assertEqual(len(parsed), 1)
            self.assertIn(
                "deployment/vllm-deploy -n prod --containers=vllm-server",
                " ".join(parsed[0]["k8s_remediation_cmds"]),
            )

    def test_missing_snapshot_file_raises_collector_error(self):
        with self.assertRaises(CollectorError):
            load_cluster_from_file("/nonexistent/kubevram-snapshot-99999.json")


class TestScalingPerformance(unittest.TestCase):
    def test_synthetic_3000_nodes_24000_gpus_24000_pods_performance(self):
        # Build synthetic large-scale state matching benchmark dimensions:
        # 3,000 nodes, 24,000 GPUs (8/node), and 24,000 unhealthy pods
        nodes = []
        pods = []
        for n_idx in range(3000):
            node_name = f"gpu-node-{n_idx:04d}"
            gpus = []
            for g_idx in range(8):
                pod_name = f"pod-{n_idx:04d}-{g_idx}"
                gpus.append(
                    GPUDevice(
                        index=str(g_idx),
                        uuid=f"GPU-{n_idx:04d}-{g_idx}",
                        model="NVIDIA H100 80GB HBM3",
                        total_vram_mib=81920,
                        used_vram_mib=81200,
                        gpu_util_pct=95.0,
                        sm_active_pct=85.0,
                        dram_active_pct=88.0,
                        temp_c=68,
                        power_w=620,
                        allocated_pods=[f"ns/{pod_name}"],
                        measured_at="2026-11-04T01:20:00Z",
                    )
                )
                pods.append(
                    InferencePod(
                        name=pod_name,
                        namespace="ns",
                        node_name=node_name,
                        workload_engine="vllm",
                        model_name="llama",
                        status="CrashLoopBackOff",
                        restarts=3,
                        last_exit_code=1,
                        last_termination_reason="Error",
                        requested_gpu_resource="nvidia.com/gpu: 1",
                        mem_request_gi=32.0,
                        mem_limit_gi=64.0,
                        mem_used_gi=24.0,
                        shm_size_mib=16384,
                        gpu_indices=[str(g_idx)],
                        cli_args={"--max-model-len": "131072"},
                        recent_logs=["torch.cuda.OutOfMemoryError: CUDA out of memory"],
                        pod_group=f"pg-{n_idx:04d}",
                        owner_kind="Deployment",
                        owner_name=f"dep-{n_idx:04d}",
                        container_name="vllm",
                        last_terminated_at="2026-11-04T01:19:50Z",
                        metrics_measured_at="2026-11-04T01:20:00Z",
                    )
                )
            nodes.append(
                ClusterNode(
                    name=node_name,
                    gpu_model="NVIDIA-H100-80GB-HBM3",
                    physical_gpus=8,
                    advertised_gpu_resources={"nvidia.com/gpu": 8},
                    sharing_strategy="none",
                    mig_strategy="none",
                    replicas_per_gpu=1,
                    interconnect_fabric="NVSwitch",
                    numa_policy="single-numa-node",
                    driver_version="550.90.07",
                    cuda_version="12.4",
                    runtime_class="nvidia",
                    gpus=gpus,
                )
            )

        t0 = time.perf_counter()
        findings = triage_cluster(nodes, pods)
        triage_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(len(findings), 24000)
        self.assertLess(triage_ms, 350.0, f"Triage regression: took {triage_ms:.1f} ms")

        t1 = time.perf_counter()
        grouped = group_pod_groups_by_node(pods)
        fabric_ms = (time.perf_counter() - t1) * 1000.0
        self.assertEqual(len(grouped), 3000)
        self.assertLess(fabric_ms, 50.0, f"Fabric grouping regression: took {fabric_ms:.1f} ms")

        t2 = time.perf_counter()
        serialized = [n.to_dict() for n in nodes]
        serial_ms = (time.perf_counter() - t2) * 1000.0
        self.assertEqual(len(serialized), 3000)
        self.assertLess(serial_ms, 150.0, f"Node serialization regression: took {serial_ms:.1f} ms")


if __name__ == "__main__":
    unittest.main()
