"""
Zero-dependency Live Cluster & Snapshot Collectors for KubeVRAM.

Supports three explicit data paths:
1. Live Kubernetes + Prometheus/DCGM collection (`kubectl` JSON + `urllib.request`).
2. Offline cluster snapshot / raw `kubectl` JSON bundle files (`--from-file`).
3. Built-in simulation fixtures (`mock`) for offline sizing and exploration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from kubevram.mock_cluster import get_mock_cluster_state
from kubevram.models import ClusterNode, ContainerState, GPUDevice, InferencePod


class CollectorError(RuntimeError):
    """Raised when live cluster or snapshot collection fails."""


_K8S_QUANTITY_FACTORS: Dict[str, float] = {
    "": 1.0,
    "Ki": 1024.0,
    "Mi": 1024.0**2,
    "Gi": 1024.0**3,
    "Ti": 1024.0**4,
    "Pi": 1024.0**5,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "m": 1e-3,
}

_QUANTITY_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")
_PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)(?:\s+(\d+))?\s*$"
)
_PROM_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_k8s_quantity_bytes(val: Any) -> int:
    """Parses a Kubernetes resource quantity string (e.g., '16Gi', '64Mi', '1000m') into bytes."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return 0
    m = _QUANTITY_RE.match(s)
    if not m:
        return 0
    num = float(m.group(1))
    suffix = m.group(2)
    factor = _K8S_QUANTITY_FACTORS.get(suffix)
    if factor is None:
        return 0
    return int(num * factor)


def _unix_to_iso(ts_seconds: float) -> str:
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_dcgm_text_metrics(metrics_text: str) -> List[Dict[str, Any]]:
    """
    Parses Prometheus / DCGM Exporter text exposition metrics into normalized sample dicts:
    {'metric': str, 'labels': dict, 'value': float, 'timestamp_iso': Optional[str]}
    """
    samples: List[Dict[str, Any]] = []
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        metric_name, labels_str, val_str, ts_ms_str = m.groups()
        try:
            value = float(val_str)
        except ValueError:
            continue
        labels: Dict[str, str] = {}
        if labels_str:
            for lm in _PROM_LABEL_RE.finditer(labels_str):
                labels[lm.group(1)] = lm.group(2)
        ts_iso = _unix_to_iso(int(ts_ms_str) / 1000.0) if ts_ms_str else None
        samples.append(
            {
                "metric": metric_name,
                "labels": labels,
                "value": value,
                "timestamp_iso": ts_iso,
            }
        )
    return samples


def _resolve_wildcard_secret_subjects(rbac_items: List[Dict[str, Any]]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Evaluates Kubernetes Role, ClusterRole, RoleBinding, and ClusterRoleBinding items to identify:
    - cluster_wide_sas: Set of 'namespace/serviceaccount' strings with cluster-wide wildcard secret read access.
    - namespaced_sas: Set of (namespace, serviceaccount) tuples with namespace-wide wildcard secret read access.
    A role has wildcard secret access if it grants '*', 'get', 'list', or 'watch' on resource '*' or 'secrets'
    without restricting 'resourceNames'.
    """
    wildcard_cluster_roles: Set[str] = set()
    wildcard_roles: Set[Tuple[str, str]] = set()
    secret_verbs = {"*", "get", "list", "watch"}

    def _rules_grant_wildcard_secrets(rules: List[Dict[str, Any]]) -> bool:
        for rule in rules or []:
            api_groups = set(rule.get("apiGroups") or [""])
            if "" not in api_groups and "*" not in api_groups:
                continue
            resources = set(rule.get("resources") or [])
            if "secrets" not in resources and "*" not in resources:
                continue
            verbs = set(rule.get("verbs") or [])
            if not (verbs & secret_verbs):
                continue
            resource_names = rule.get("resourceNames") or []
            if not resource_names:
                return True
        return False

    for item in rbac_items:
        kind = item.get("kind")
        meta = item.get("metadata") or {}
        name = meta.get("name", "")
        ns = meta.get("namespace", "")
        if kind == "ClusterRole" and _rules_grant_wildcard_secrets(item.get("rules") or []):
            wildcard_cluster_roles.add(name)
        elif kind == "Role" and _rules_grant_wildcard_secrets(item.get("rules") or []):
            wildcard_roles.add((ns, name))

    cluster_wide_sas: Set[str] = set()
    namespaced_sas: Set[Tuple[str, str]] = set()

    for item in rbac_items:
        kind = item.get("kind")
        meta = item.get("metadata") or {}
        binding_ns = meta.get("namespace", "")
        role_ref = item.get("roleRef") or {}
        ref_kind = role_ref.get("kind")
        ref_name = role_ref.get("name", "")
        subjects = item.get("subjects") or []

        is_wildcard_ref = (
            (ref_kind == "ClusterRole" and ref_name in wildcard_cluster_roles)
            or (ref_kind == "Role" and (binding_ns, ref_name) in wildcard_roles)
        )
        if not is_wildcard_ref:
            continue

        for subj in subjects:
            if subj.get("kind") == "ServiceAccount":
                sa_name = subj.get("name", "default")
                sa_ns = subj.get("namespace") or binding_ns or "default"
                if kind == "ClusterRoleBinding":
                    cluster_wide_sas.add(f"{sa_ns}/{sa_name}")
                elif kind == "RoleBinding" and binding_ns:
                    namespaced_sas.add((binding_ns, sa_name))

    return cluster_wide_sas, namespaced_sas


def _parse_cli_flags_from_container(container_spec: Dict[str, Any]) -> Dict[str, str]:
    tokens: List[str] = []
    for key in ("command", "args"):
        val = container_spec.get(key)
        if isinstance(val, list):
            tokens.extend(str(x) for x in val)
    cli_args: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                cli_args[k] = v
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                cli_args[tok] = tokens[i + 1]
                i += 1
            else:
                cli_args[tok] = "true"
        i += 1
    return cli_args


def _detect_workload_engine(image: str, cli_args: Dict[str, str]) -> Tuple[str, str]:
    img_lower = (image or "").lower()
    model_name = cli_args.get("--model") or cli_args.get("--model-repository") or "unknown"
    if "vllm" in img_lower:
        return "vllm", model_name
    if "triton" in img_lower:
        return "triton", model_name
    if "kserve" in img_lower:
        return "kserve", model_name
    if "ray" in img_lower:
        return "ray-worker", model_name
    return "pytorch-loader", model_name


def parse_k8s_objects_to_state(
    node_items: List[Dict[str, Any]],
    pod_items: List[Dict[str, Any]],
    rs_items: Optional[List[Dict[str, Any]]] = None,
    rbac_items: Optional[List[Dict[str, Any]]] = None,
    dcgm_samples: Optional[List[Dict[str, Any]]] = None,
    pod_logs_by_key: Optional[Dict[str, List[str]]] = None,
    pod_mem_used_bytes: Optional[Dict[str, int]] = None,
    scrape_timestamp: Optional[str] = None,
) -> Tuple[List[ClusterNode], List[InferencePod]]:
    """
    Transforms raw Kubernetes API Node/Pod/ReplicaSet/RBAC objects and DCGM metric samples
    into KubeVRAM domain models (`ClusterNode`, `InferencePod`).
    """
    measured_ts = scrape_timestamp or _now_iso()
    rs_owner_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for rs in rs_items or []:
        meta = rs.get("metadata") or {}
        rs_ns = meta.get("namespace", "default")
        rs_name = meta.get("name", "")
        for oref in meta.get("ownerReferences") or []:
            if oref.get("kind") and oref.get("name"):
                rs_owner_map[(rs_ns, rs_name)] = (oref["kind"], oref["name"])
                break

    cluster_wide_sas, namespaced_sas = _resolve_wildcard_secret_subjects(rbac_items or [])

    # Group DCGM samples by (node_name, gpu_key)
    dcgm_by_node_gpu: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sample in dcgm_samples or []:
        lbls = sample["labels"]
        node_name = (
            lbls.get("Hostname")
            or lbls.get("kubernetes_node")
            or lbls.get("node")
            or ""
        )
        mig_prof = lbls.get("GPU_I_PROFILE")
        gpu_id = lbls.get("gpu") or lbls.get("device") or "0"
        gpu_i_id = lbls.get("GPU_I_ID")
        gpu_key = f"{gpu_id}/MIG-{gpu_i_id} ({mig_prof})" if (mig_prof and gpu_i_id is not None) else gpu_id

        entry = dcgm_by_node_gpu.setdefault(
            (node_name, gpu_key),
            {
                "index": gpu_key,
                "uuid": lbls.get("UUID") or f"GPU-{node_name}-{gpu_key}",
                "model": lbls.get("modelName") or "",
                "mig_profile": mig_prof,
                "metrics": {},
                "pods": set(),
                "measured_at": sample.get("timestamp_iso") or measured_ts,
            },
        )
        entry["metrics"][sample["metric"]] = sample["value"]
        exp_ns = lbls.get("exported_namespace") or lbls.get("namespace")
        exp_pod = lbls.get("exported_pod") or lbls.get("pod")
        if exp_ns and exp_pod:
            entry["pods"].add(f"{exp_ns}/{exp_pod}")

    nodes: List[ClusterNode] = []
    nodes_by_name: Dict[str, ClusterNode] = {}

    for item in node_items:
        meta = item.get("metadata") or {}
        labels = meta.get("labels") or {}
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        allocatable = status.get("allocatable") or {}

        adv_gpu: Dict[str, int] = {}
        for res_k, res_v in allocatable.items():
            if res_k.startswith("nvidia.com/gpu") or res_k.startswith("nvidia.com/mig-"):
                try:
                    cnt = int(str(res_v))
                except ValueError:
                    cnt = 0
                if cnt > 0:
                    adv_gpu[res_k] = cnt

        has_gpu_label = labels.get("nvidia.com/gpu.present") == "true" or bool(adv_gpu)
        if not has_gpu_label:
            continue

        node_name = meta.get("name", "unknown-node")
        gpu_model = labels.get("nvidia.com/gpu.product", "NVIDIA-GPU")
        phys_gpus = int(labels.get("nvidia.com/gpu.count", sum(adv_gpu.values()) or 1))
        vram_mib_label = int(labels.get("nvidia.com/gpu.memory", 81920))
        sharing_strategy = labels.get("nvidia.com/gpu.sharing-strategy", "none").lower()
        mig_strategy = labels.get("nvidia.com/mig.strategy", "none").lower()
        if mig_strategy != "none" and sharing_strategy == "none":
            sharing_strategy = "mig"
        replicas = int(labels.get("nvidia.com/gpu.replicas", 1))

        drv_ver = labels.get(
            "nvidia.com/cuda.driver-version.full",
            f"{labels.get('nvidia.com/cuda.driver.major', '550')}.{labels.get('nvidia.com/cuda.driver.minor', '0')}",
        )
        cuda_ver = labels.get(
            "nvidia.com/cuda.runtime-version.full",
            f"{labels.get('nvidia.com/cuda.runtime.major', '12')}.{labels.get('nvidia.com/cuda.runtime.minor', '4')}",
        )
        numa_policy = labels.get("kubevram.io/numa-policy") or labels.get(
            "topology.kubernetes.io/numa-policy", "single-numa-node"
        )
        cdi_enabled = (
            labels.get("nvidia.com/gpu.cdi") == "true"
            or labels.get("nvidia.com/cdi.enabled") == "true"
        )
        default_rt_nvidia = labels.get("nvidia.com/gpu.default-runtime") == "true"
        rt_class = labels.get("nvidia.com/gpu.runtime-class", "nvidia")
        interconnect = labels.get("nvidia.com/gpu.fabric", "PCIe / NVLink")
        schedulable = not bool(spec.get("unschedulable", False))

        node_gpus: List[GPUDevice] = []
        matched_dcgm = [
            v for (n_name, _), v in sorted(dcgm_by_node_gpu.items()) if n_name == node_name
        ]
        if matched_dcgm:
            for d in matched_dcgm:
                m = d["metrics"]
                used_mib = int(round(m.get("DCGM_FI_DEV_FB_USED", 0.0)))
                free_mib = int(round(m.get("DCGM_FI_DEV_FB_FREE", 0.0)))
                total_mib = (used_mib + free_mib) if (used_mib + free_mib) > 0 else vram_mib_label
                sm_raw = m.get("DCGM_FI_PROF_SM_ACTIVE", 0.0)
                sm_pct = round(sm_raw * 100.0 if sm_raw <= 1.0 else sm_raw, 1)
                dram_raw = m.get("DCGM_FI_PROF_DRAM_ACTIVE", 0.0)
                dram_pct = round(dram_raw * 100.0 if dram_raw <= 1.0 else dram_raw, 1)
                xid_val = int(m.get("DCGM_FI_DEV_XID_ERRORS", 0))
                node_gpus.append(
                    GPUDevice(
                        index=d["index"],
                        uuid=d["uuid"],
                        model=d["model"] or gpu_model,
                        total_vram_mib=total_mib,
                        used_vram_mib=used_mib,
                        gpu_util_pct=round(float(m.get("DCGM_FI_DEV_GPU_UTIL", 0.0)), 1),
                        sm_active_pct=sm_pct,
                        dram_active_pct=dram_pct,
                        temp_c=int(round(m.get("DCGM_FI_DEV_GPU_TEMP", 0.0))),
                        power_w=int(round(m.get("DCGM_FI_DEV_POWER_USAGE", 0.0))),
                        xid_error=xid_val,
                        mig_profile=d["mig_profile"],
                        interconnect=interconnect,
                        allocated_pods=sorted(d["pods"]),
                        measured_at=d["measured_at"],
                        xid_timestamp=d["measured_at"] if xid_val > 0 else None,
                    )
                )
        else:
            for i in range(phys_gpus):
                node_gpus.append(
                    GPUDevice(
                        index=str(i),
                        uuid=f"GPU-{node_name}-{i}",
                        model=gpu_model,
                        total_vram_mib=vram_mib_label,
                        used_vram_mib=0,
                        gpu_util_pct=0.0,
                        sm_active_pct=0.0,
                        dram_active_pct=0.0,
                        temp_c=0,
                        power_w=0,
                        xid_error=0,
                        interconnect=interconnect,
                        measured_at=None,
                    )
                )

        cnode = ClusterNode(
            name=node_name,
            gpu_model=gpu_model,
            physical_gpus=phys_gpus,
            advertised_gpu_resources=adv_gpu or {"nvidia.com/gpu": phys_gpus},
            sharing_strategy=sharing_strategy,
            mig_strategy=mig_strategy,
            replicas_per_gpu=replicas,
            interconnect_fabric=interconnect,
            numa_policy=numa_policy,
            driver_version=drv_ver,
            cuda_version=cuda_ver,
            runtime_class=rt_class,
            schedulable=schedulable,
            gpus=node_gpus,
            default_runtime_is_nvidia=default_rt_nvidia,
            cdi_enabled=cdi_enabled,
            measured_at=measured_ts if matched_dcgm else None,
        )
        nodes.append(cnode)
        nodes_by_name[node_name] = cnode

    pods: List[InferencePod] = []
    for p_item in pod_items:
        meta = p_item.get("metadata") or {}
        spec = p_item.get("spec") or {}
        status = p_item.get("status") or {}
        ns = meta.get("namespace", "default")
        pod_name = meta.get("name", "unknown-pod")
        node_name = spec.get("nodeName") or ""

        containers_spec = spec.get("containers") or []
        gpu_reqs: List[str] = []
        total_mem_req_bytes = 0
        total_mem_lim_bytes = 0
        privileged = bool((spec.get("securityContext") or {}).get("privileged", False))
        primary_cspec = containers_spec[0] if containers_spec else {}

        for cspec in containers_spec:
            res = cspec.get("resources") or {}
            reqs = res.get("requests") or {}
            lims = res.get("limits") or {}
            merged_gpu = {
                k: lims.get(k, reqs.get(k))
                for k in set(reqs.keys()) | set(lims.keys())
                if k.startswith("nvidia.com/") or k == "amd.com/gpu"
            }
            for gk, gv in sorted(merged_gpu.items()):
                if str(gv) != "0":
                    gpu_reqs.append(f"{gk}: {gv}")
                    primary_cspec = cspec
            total_mem_req_bytes += parse_k8s_quantity_bytes(reqs.get("memory"))
            total_mem_lim_bytes += parse_k8s_quantity_bytes(lims.get("memory"))
            if (cspec.get("securityContext") or {}).get("privileged") is True:
                privileged = True

        if not gpu_reqs:
            continue

        # Parse /dev/shm mount size
        volumes_by_name = {v.get("name"): v for v in (spec.get("volumes") or []) if v.get("name")}
        shm_size_mib = 64
        for cspec in containers_spec:
            for vm in cspec.get("volumeMounts") or []:
                if vm.get("mountPath") == "/dev/shm":
                    vol = volumes_by_name.get(vm.get("name"), {})
                    empty_dir = vol.get("emptyDir")
                    if isinstance(empty_dir, dict) and empty_dir.get("medium") == "Memory":
                        limit_bytes = parse_k8s_quantity_bytes(empty_dir.get("sizeLimit"))
                        shm_size_mib = (
                            max(1, limit_bytes // (1024 * 1024))
                            if limit_bytes > 0
                            else max(64, int((total_mem_lim_bytes // (1024 * 1024)) // 2) or 8192)
                        )

        # Resolve owner controller (unwrapping ReplicaSet -> Deployment)
        owner_kind: Optional[str] = None
        owner_name: Optional[str] = None
        for oref in meta.get("ownerReferences") or []:
            okind = oref.get("kind")
            oname = oref.get("name")
            if okind == "ReplicaSet" and oname and (ns, oname) in rs_owner_map:
                owner_kind, owner_name = rs_owner_map[(ns, oname)]
            elif okind and oname:
                owner_kind, owner_name = okind, oname
            break

        # Parse containerStatuses
        c_states: List[ContainerState] = []
        total_restarts = 0
        last_exit = 0
        last_reason = "None"
        last_term_at: Optional[str] = None
        effective_status = status.get("phase", "Unknown")

        for cstat in status.get("containerStatuses") or []:
            cname = cstat.get("name", "container")
            restarts = int(cstat.get("restartCount", 0))
            total_restarts += restarts
            ready = bool(cstat.get("ready", False))
            st_obj = cstat.get("state") or {}
            last_obj = (cstat.get("lastState") or {}).get("terminated") or {}

            c_state_label = "running"
            wait_reason: Optional[str] = None
            exit_code = int(last_obj.get("exitCode", 0))
            term_reason = last_obj.get("reason") or "None"
            term_at = last_obj.get("finishedAt")

            if "waiting" in st_obj:
                c_state_label = "waiting"
                wait_reason = (st_obj["waiting"] or {}).get("reason")
                if wait_reason:
                    effective_status = wait_reason
            elif "terminated" in st_obj:
                c_state_label = "terminated"
                cur_term = st_obj["terminated"] or {}
                exit_code = int(cur_term.get("exitCode", exit_code))
                term_reason = cur_term.get("reason") or term_reason
                term_at = cur_term.get("finishedAt") or term_at
                if term_reason == "OOMKilled":
                    effective_status = "OOMKilled"
                elif exit_code != 0 and effective_status not in ("CrashLoopBackOff", "OOMKilled"):
                    effective_status = "Error"

            if exit_code != 0 or last_exit == 0:
                last_exit = exit_code
            if term_reason not in ("None", ""):
                last_reason = term_reason
            if term_at:
                last_term_at = term_at

            c_states.append(
                ContainerState(
                    name=cname,
                    state=c_state_label,
                    ready=ready,
                    restarts=restarts,
                    exit_code=exit_code,
                    termination_reason=term_reason,
                    waiting_reason=wait_reason,
                    terminated_at=term_at,
                )
            )

        cli_args = _parse_cli_flags_from_container(primary_cspec)
        engine, model_name = _detect_workload_engine(primary_cspec.get("image", ""), cli_args)
        annotations = meta.get("annotations") or {}
        p_labels = meta.get("labels") or {}
        pod_group = (
            annotations.get("scheduling.k8s.io/group-name")
            or annotations.get("scheduling.volcano.sh/group-name")
            or p_labels.get("pod-group")
        )
        sa_name = spec.get("serviceAccountName") or spec.get("serviceAccount") or "default"
        has_wildcard = (f"{ns}/{sa_name}" in cluster_wide_sas) or ((ns, sa_name) in namespaced_sas)

        pod_key = f"{ns}/{pod_name}"
        assigned_gpu_indices: List[str] = []
        cnode = nodes_by_name.get(node_name)
        if cnode:
            for g in cnode.gpus:
                if pod_key in g.allocated_pods:
                    assigned_gpu_indices.append(g.index)

        mem_used_bytes = (pod_mem_used_bytes or {}).get(pod_key, 0)
        mem_req_gi = round(total_mem_req_bytes / (1024**3), 2)
        mem_lim_gi = round(total_mem_lim_bytes / (1024**3), 2)
        mem_used_gi = round(mem_used_bytes / (1024**3), 2)

        pods.append(
            InferencePod(
                name=pod_name,
                namespace=ns,
                node_name=node_name,
                workload_engine=engine,
                model_name=model_name,
                status=effective_status,
                restarts=total_restarts,
                last_exit_code=last_exit,
                last_termination_reason=last_reason,
                requested_gpu_resource=", ".join(gpu_reqs),
                mem_request_gi=mem_req_gi,
                mem_limit_gi=mem_lim_gi,
                mem_used_gi=mem_used_gi,
                shm_size_mib=shm_size_mib,
                gpu_indices=assigned_gpu_indices,
                cli_args=cli_args,
                recent_logs=(pod_logs_by_key or {}).get(pod_key, []),
                service_account=sa_name,
                privileged=privileged,
                runtime_class_name=spec.get("runtimeClassName"),
                pod_group=pod_group,
                wildcard_secrets=has_wildcard,
                owner_kind=owner_kind,
                owner_name=owner_name,
                container_name=primary_cspec.get("name"),
                containers=c_states,
                last_terminated_at=last_term_at,
                metrics_measured_at=measured_ts if (dcgm_samples or pod_mem_used_bytes) else None,
            )
        )

    return nodes, pods


def _deserialize_snapshot_dict(raw: Dict[str, Any]) -> Tuple[List[ClusterNode], List[InferencePod]]:
    """Loads either a KubeVRAM model snapshot dict or a raw K8s API bundle dict."""
    raw_nodes = raw.get("nodes") or []
    raw_pods = raw.get("pods") or []

    # Detect raw K8s API objects vs serialized KubeVRAM dataclasses
    if isinstance(raw_nodes, dict) and raw_nodes.get("kind", "").endswith("List"):
        node_items = raw_nodes.get("items") or []
        pod_items = (raw_pods.get("items") if isinstance(raw_pods, dict) else raw_pods) or []
        rs_items = (raw.get("replicasets", {}).get("items") if isinstance(raw.get("replicasets"), dict) else raw.get("replicasets")) or []
        rbac_items = raw.get("rbac") or []
        dcgm_text = raw.get("dcgm_metrics") or ""
        dcgm_samples = parse_dcgm_text_metrics(dcgm_text) if isinstance(dcgm_text, str) else raw.get("dcgm_samples")
        return parse_k8s_objects_to_state(
            node_items=node_items,
            pod_items=pod_items,
            rs_items=rs_items,
            rbac_items=rbac_items,
            dcgm_samples=dcgm_samples,
            pod_logs_by_key=raw.get("pod_logs"),
            scrape_timestamp=raw.get("measured_at"),
        )

    if raw_nodes and isinstance(raw_nodes[0], dict) and "apiVersion" in raw_nodes[0]:
        return parse_k8s_objects_to_state(
            node_items=raw_nodes,
            pod_items=raw_pods,
            rs_items=raw.get("replicasets"),
            rbac_items=raw.get("rbac"),
            dcgm_samples=parse_dcgm_text_metrics(raw["dcgm_metrics"]) if isinstance(raw.get("dcgm_metrics"), str) else None,
            pod_logs_by_key=raw.get("pod_logs"),
            scrape_timestamp=raw.get("measured_at"),
        )

    # Serialized KubeVRAM ClusterNode / InferencePod dictionaries
    nodes: List[ClusterNode] = []
    for nd in raw_nodes:
        gpus: List[GPUDevice] = []
        for gd in nd.get("gpus") or []:
            gd_clean = {
                k: v
                for k, v in gd.items()
                if k not in ("free_vram_mib", "vram_util_pct")
            }
            gpus.append(GPUDevice(**gd_clean))
        nd_clean = {k: v for k, v in nd.items() if k != "gpus"}
        nodes.append(ClusterNode(**nd_clean, gpus=gpus))

    pods: List[InferencePod] = []
    for pd in raw_pods:
        containers = [ContainerState(**cd) for cd in (pd.get("containers") or [])]
        pd_clean = {k: v for k, v in pd.items() if k != "containers"}
        pods.append(InferencePod(**pd_clean, containers=containers))

    return nodes, pods


def load_cluster_from_file(path: str) -> Tuple[List[ClusterNode], List[InferencePod]]:
    """Loads cluster state from a JSON snapshot or raw Kubernetes API bundle file."""
    if not os.path.isfile(path):
        raise CollectorError(f"Snapshot file not found: '{path}'")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise CollectorError(f"Failed to parse JSON snapshot '{path}': {exc}") from exc

    if isinstance(data, dict) and data.get("kind") == "List":
        items = data.get("items") or []
        node_items = [i for i in items if i.get("kind") == "Node"]
        pod_items = [i for i in items if i.get("kind") == "Pod"]
        rs_items = [i for i in items if i.get("kind") == "ReplicaSet"]
        rbac_items = [
            i for i in items if i.get("kind") in ("Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding")
        ]
        return parse_k8s_objects_to_state(node_items, pod_items, rs_items=rs_items, rbac_items=rbac_items)

    if not isinstance(data, dict):
        raise CollectorError(f"Expected JSON object in '{path}', got {type(data).__name__}")
    return _deserialize_snapshot_dict(data)


def _run_kubectl_json(base_cmd: List[str], sub_args: List[str], timeout: int = 15) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            base_cmd + sub_args + ["-o", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CollectorError("kubectl binary not found on PATH; cannot query live cluster.") from exc
    except subprocess.TimeoutExpired as exc:
        raise CollectorError(f"kubectl timed out after {timeout}s while running: {' '.join(sub_args)}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise CollectorError(f"kubectl {' '.join(sub_args)} failed (exit {proc.returncode}): {stderr}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CollectorError(f"Invalid JSON returned by kubectl {' '.join(sub_args)}: {exc}") from exc


def _query_prometheus_dcgm(prom_url: str, timeout: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Queries Prometheus HTTP API (/api/v1/query) for DCGM GPU metrics and cAdvisor pod RAM."""
    dcgm_query = (
        '{__name__=~"DCGM_FI_DEV_FB_USED|DCGM_FI_DEV_FB_FREE|DCGM_FI_DEV_GPU_UTIL|'
        'DCGM_FI_PROF_SM_ACTIVE|DCGM_FI_PROF_DRAM_ACTIVE|DCGM_FI_DEV_GPU_TEMP|'
        'DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_XID_ERRORS"}'
    )
    samples: List[Dict[str, Any]] = []
    mem_by_pod: Dict[str, int] = {}

    def _prom_get(expr: str) -> List[Dict[str, Any]]:
        url = f"{prom_url.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("status") != "success":
            raise CollectorError(f"Prometheus query failed: {payload}")
        return (payload.get("data") or {}).get("result") or []

    try:
        for res in _prom_get(dcgm_query):
            metric_labels = dict(res.get("metric") or {})
            metric_name = metric_labels.pop("__name__", "")
            val_pair = res.get("value") or [0, "0"]
            ts_iso = _unix_to_iso(float(val_pair[0])) if val_pair and val_pair[0] else None
            samples.append(
                {
                    "metric": metric_name,
                    "labels": metric_labels,
                    "value": float(val_pair[1]),
                    "timestamp_iso": ts_iso,
                }
            )
        for res in _prom_get('sum by (namespace, pod) (container_memory_working_set_bytes{container!="",container!="POD"})'):
            ml = res.get("metric") or {}
            ns = ml.get("namespace")
            pod = ml.get("pod")
            val_pair = res.get("value") or [0, "0"]
            if ns and pod:
                mem_by_pod[f"{ns}/{pod}"] = int(float(val_pair[1]))
    except CollectorError:
        raise
    except Exception as exc:
        raise CollectorError(f"Failed querying Prometheus at '{prom_url}': {exc}") from exc

    return samples, mem_by_pod


def _scrape_dcgm_exporter(dcgm_url: str, timeout: int = 10) -> List[Dict[str, Any]]:
    try:
        req = urllib.request.Request(dcgm_url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise CollectorError(f"Failed scraping DCGM Exporter at '{dcgm_url}': {exc}") from exc
    return parse_dcgm_text_metrics(text)


def collect_live_cluster_state(
    kubeconfig: Optional[str] = None,
    context: Optional[str] = None,
    namespace: Optional[str] = None,
    prom_url: Optional[str] = None,
    dcgm_url: Optional[str] = None,
) -> Tuple[List[ClusterNode], List[InferencePod]]:
    """Collects live GPU cluster state from Kubernetes API and optional Prometheus / DCGM endpoints."""
    kubectl_bin = shutil.which("kubectl")
    if not kubectl_bin:
        raise CollectorError("kubectl is not installed or not on PATH; cannot run with --live.")

    base_cmd = [kubectl_bin]
    if kubeconfig:
        base_cmd.extend(["--kubeconfig", kubeconfig])
    if context:
        base_cmd.extend(["--context", context])

    ns_args = ["-n", namespace] if namespace else ["-A"]
    scrape_ts = _now_iso()

    nodes_doc = _run_kubectl_json(base_cmd, ["get", "nodes"])
    pods_doc = _run_kubectl_json(base_cmd, ["get", "pods"] + ns_args)

    rs_items: List[Dict[str, Any]] = []
    try:
        rs_doc = _run_kubectl_json(base_cmd, ["get", "replicasets"] + ns_args)
        rs_items = rs_doc.get("items") or []
    except CollectorError:
        pass

    rbac_items: List[Dict[str, Any]] = []
    for kind, scoped_args in (
        ("roles", ns_args),
        ("rolebindings", ns_args),
        ("clusterroles", []),
        ("clusterrolebindings", []),
    ):
        try:
            doc = _run_kubectl_json(base_cmd, ["get", kind] + scoped_args)
            resolved_kind = {
                "roles": "Role",
                "rolebindings": "RoleBinding",
                "clusterroles": "ClusterRole",
                "clusterrolebindings": "ClusterRoleBinding",
            }[kind]
            for it in doc.get("items") or []:
                it.setdefault("kind", resolved_kind)
                rbac_items.append(it)
        except CollectorError:
            # RBAC read permission may not be granted to the caller; skip gracefully
            pass

    dcgm_samples: List[Dict[str, Any]] = []
    mem_by_pod: Dict[str, int] = {}
    if prom_url:
        dcgm_samples, mem_by_pod = _query_prometheus_dcgm(prom_url)
    elif dcgm_url:
        dcgm_samples = _scrape_dcgm_exporter(dcgm_url)

    # Fetch recent logs for unhealthy GPU pods
    pod_logs: Dict[str, List[str]] = {}
    for p_item in pods_doc.get("items") or []:
        meta = p_item.get("metadata") or {}
        spec = p_item.get("spec") or {}
        status = p_item.get("status") or {}
        has_gpu = any(
            any(k.startswith("nvidia.com/") for k in ((c.get("resources") or {}).get("limits") or {}))
            or any(k.startswith("nvidia.com/") for k in ((c.get("resources") or {}).get("requests") or {}))
            for c in (spec.get("containers") or [])
        )
        if not has_gpu:
            continue
        c_statuses = status.get("containerStatuses") or []
        unhealthy = status.get("phase") not in ("Running", "Succeeded") or any(
            int(cs.get("restartCount", 0)) > 0 or not cs.get("ready", True) for cs in c_statuses
        )
        if not unhealthy:
            continue
        p_ns = meta.get("namespace", "default")
        p_name = meta.get("name", "")
        c_name = (spec.get("containers") or [{}])[0].get("name")
        for prev_flag in (["--previous"], []):
            log_cmd = base_cmd + ["logs", "-n", p_ns, p_name, "--tail=40"] + prev_flag
            if c_name:
                log_cmd.extend(["-c", c_name])
            proc = subprocess.run(log_cmd, capture_output=True, text=True, timeout=8, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                pod_logs[f"{p_ns}/{p_name}"] = proc.stdout.strip().splitlines()[-40:]
                break

    return parse_k8s_objects_to_state(
        node_items=nodes_doc.get("items") or [],
        pod_items=pods_doc.get("items") or [],
        rs_items=rs_items,
        rbac_items=rbac_items,
        dcgm_samples=dcgm_samples,
        pod_logs_by_key=pod_logs,
        pod_mem_used_bytes=mem_by_pod,
        scrape_timestamp=scrape_ts,
    )


def resolve_cluster_state(
    live: bool = False,
    from_file: Optional[str] = None,
    kubeconfig: Optional[str] = None,
    context: Optional[str] = None,
    namespace: Optional[str] = None,
    prom_url: Optional[str] = None,
    dcgm_url: Optional[str] = None,
) -> Tuple[str, List[ClusterNode], List[InferencePod]]:
    """
    Resolves cluster telemetry state from the requested source and returns
    (source_label, nodes, pods).
    """
    env_source = os.environ.get("KUBEVRAM_SOURCE", "").strip().lower()
    env_file = os.environ.get("KUBEVRAM_SNAPSHOT_FILE", "").strip() or None

    target_file = from_file or env_file
    if target_file:
        nodes, pods = load_cluster_from_file(target_file)
        return f"file:{target_file}", nodes, pods

    if live or env_source == "live":
        nodes, pods = collect_live_cluster_state(
            kubeconfig=kubeconfig or os.environ.get("KUBECONFIG"),
            context=context,
            namespace=namespace,
            prom_url=prom_url or os.environ.get("KUBEVRAM_PROM_URL"),
            dcgm_url=dcgm_url or os.environ.get("KUBEVRAM_DCGM_URL"),
        )
        return "live", nodes, pods

    nodes, pods = get_mock_cluster_state()
    return "mock-simulation", nodes, pods
