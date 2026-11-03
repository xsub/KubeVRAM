"""
KubeVRAM CLI Entrypoint and Subcommand Implementations.

Subcommands:
- topology: GPU nodes, sharing strategy, per-GPU VRAM bars, SM utilization, NVLink/PCIe fabric.
- triage: Automated OOM (Exit 137 vs CUDA Exit 1 vs Time-Slicing vs XID) Root Cause Analysis.
- tune-vllm: Deterministic KV-cache & VRAM sizing calculator + deployment manifest generator.
- partition: Hardware-aware MIG vs MPS vs Time-Slicing advisor + ConfigMap generator.
- bridge: Linux-to-Kubernetes command & intent translation engine.
- rbac-audit: Cluster GPU RBAC, privileged container, and runtime security auditor.
- overview: Cluster health banner, node summaries, and quickstart guidance.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from kubevram.linux_bridge import translate_linux_intent
from kubevram.mock_cluster import get_mock_cluster_state
from kubevram.models import ClusterNode, GPUDevice, InferencePod, RCAFinding
from kubevram.oom_correlator import triage_cluster
from kubevram.partition_advisor import recommend_partitioning
from kubevram.rbac_auditor import audit_cluster_rbac
from kubevram.ui import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    colorize,
    render_panel,
    render_table,
    severity_badge,
    strip_ansi,
    vram_bar,
)
from kubevram.vllm_calculator import (
    GPU_CATALOG,
    MODEL_CATALOG,
    calculate_vllm_tuning,
)


def cmd_topology(args: argparse.Namespace) -> int:
    """Renders GPU node topology, sharing strategy, DCGM VRAM bars, and interconnect fabric."""
    nodes, pods = get_mock_cluster_state()

    if args.node:
        target_name = args.node.strip().lower()
        nodes = [n for n in nodes if target_name in n.name.lower()]
        if not nodes:
            print(f"{RED}Error: Node matching '{args.node}' not found.{RESET}", file=sys.stderr)
            return 1

    if args.format == "json":
        data = [n.to_dict() for n in nodes]
        print(json.dumps(data, indent=2))
        return 0

    print(f"\n{BOLD}{CYAN}=== KubeVRAM Cluster GPU Topology & Hardware Telemetry ==={RESET}\n")

    for node in nodes:
        sharing_label = node.sharing_strategy.upper()
        if node.sharing_strategy == "time-slicing":
            sharing_label += f" ({node.replicas_per_gpu}x over-subscription)"
        elif node.sharing_strategy == "mig":
            sharing_label += f" (strategy={node.mig_strategy})"

        node_info_lines = [
            f"{BOLD}Node:{RESET} {node.name}   {BOLD}Model:{RESET} {node.gpu_model}   {BOLD}Sharing:{RESET} {sharing_label}",
            f"{BOLD}Driver / CUDA:{RESET} {node.driver_version} / {node.cuda_version}   {BOLD}RuntimeClass:{RESET} {node.runtime_class}   {BOLD}Schedulable:{RESET} {node.schedulable}",
        ]

        if args.fabric:
            pod_groups = sorted(
                list(
                    {
                        p.pod_group
                        for p in pods
                        if p.node_name == node.name and p.pod_group is not None
                    }
                )
            )
            pg_str = ", ".join(pod_groups) if pod_groups else "None"
            node_info_lines.append(
                f"{BOLD}Interconnect Fabric:{RESET} {node.interconnect_fabric}   {BOLD}NUMA Policy:{RESET} {node.numa_policy}"
            )
            node_info_lines.append(f"{BOLD}Gang Scheduling (PodGroup):{RESET} {pg_str}")

        print(render_panel(f"Node: {node.name}", node_info_lines, border_color=CYAN))

        # Per-GPU table for this node
        headers = [
            "GPU",
            "Model",
            "VRAM Bar (DCGM)",
            "Used / Total",
            "SM Act %",
            "Temp / Power",
            "XID State",
            "Allocated Pods",
        ]
        rows: List[List[str]] = []

        for gpu in node.gpus:
            gpu_label = gpu.index
            if gpu.mig_profile:
                gpu_label = f"{gpu.index} ({gpu.mig_profile})"

            used_str = f"{gpu.used_vram_mib:,} / {gpu.total_vram_mib:,} MiB"
            sm_str = f"{gpu.sm_active_pct:.1f}%"
            tp_str = f"{gpu.temp_c}°C / {gpu.power_w}W"

            if gpu.xid_error > 0:
                xid_str = f"{RED}{BOLD}XID {gpu.xid_error}{RESET}"
            else:
                xid_str = f"{GREEN}OK (0){RESET}"

            pod_names = [p.split("/")[-1] for p in gpu.allocated_pods]
            alloc_str = ", ".join(pod_names) if pod_names else f"{DIM}idle{RESET}"

            rows.append(
                [
                    gpu_label,
                    gpu.model,
                    vram_bar(gpu.used_vram_mib, gpu.total_vram_mib, width=16),
                    used_str,
                    sm_str,
                    tp_str,
                    xid_str,
                    alloc_str,
                ]
            )

        print(render_table(headers, rows, title=f"GPU Devices on {node.name}"))
        print()

    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    """Correlates container terminations, host RAM, DCGM VRAM, and logs into actionable RCA."""
    nodes, pods = get_mock_cluster_state()
    findings = triage_cluster(nodes, pods, pod_filter=args.pod)

    if args.format == "json":
        data = [f.to_dict() for f in findings]
        print(json.dumps(data, indent=2))
        return 0

    print(f"\n{BOLD}{RED}=== KubeVRAM OOM & Hardware Fault Correlation Engine ==={RESET}\n")

    if not findings:
        filter_msg = f" matching '{args.pod}'" if args.pod else ""
        print(f"{GREEN}✓ No unhealthy inference pods or hardware faults detected{filter_msg}.{RESET}\n")
        return 0

    # Summary table
    headers = ["Namespace", "Pod Name", "Failure Category", "Severity", "Conf.", "Node"]
    rows: List[List[str]] = []
    for f in findings:
        rows.append(
            [
                f.namespace,
                f.pod_name,
                f.failure_category,
                severity_badge(f.severity),
                f"{int(f.confidence * 100)}%",
                f.node_name,
            ]
        )

    print(render_table(headers, rows, title="Unhealthy Workloads Triage Summary"))
    print()

    # Detailed RCA panels for each finding
    for idx, f in enumerate(findings, 1):
        panel_color = RED if f.severity in ("CRITICAL", "HIGH") else YELLOW
        title = f"RCA #{idx}: {f.namespace}/{f.pod_name} — {f.failure_category}"

        lines = [
            f"{BOLD}Summary:{RESET} {f.summary}",
            f"{BOLD}Node:{RESET} {f.node_name}   {BOLD}Confidence:{RESET} {int(f.confidence * 100)}%   {BOLD}Severity:{RESET} {severity_badge(f.severity)}",
            f"{BOLD}Root Cause:{RESET} {f.root_cause}",
            "",
            f"{BOLD}Technical Evidence:{RESET}",
        ]
        for ev in f.technical_evidence:
            lines.append(f"  • {ev}")

        lines.append("")
        lines.append(f"{BOLD}Linux Sysadmin Equivalent Command:{RESET}")
        lines.append(f"  $ {f.linux_equivalent_cmd}")

        lines.append("")
        lines.append(f"{BOLD}Kubernetes Remediation Commands:{RESET}")
        for cmd in f.k8s_remediation_cmds:
            lines.append(f"  $ {cmd}")

        if args.show_patch and f.yaml_patch:
            lines.append("")
            lines.append(f"{BOLD}Declarative YAML Remediation Patch:{RESET}")
            for patch_line in f.yaml_patch.split("\n"):
                lines.append(f"  {patch_line}")

        print(render_panel(title, lines, border_color=panel_color))
        print()

    return 0


def cmd_tune_vllm(args: argparse.Namespace) -> int:
    """Calculates deterministic VRAM budget, KV cache capacity, and vLLM CLI flags."""
    try:
        res = calculate_vllm_tuning(
            model_key=args.model,
            gpu_sku=args.gpu,
            num_gpus=args.num_gpus,
            dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            target_context=args.target_context,
            target_concurrency=args.concurrency,
        )
    except Exception as e:
        print(f"{RED}Error calculating vLLM tuning: {e}{RESET}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(res, indent=2))
        return 0

    print(f"\n{BOLD}{CYAN}=== KubeVRAM Deterministic vLLM & KV-Cache Tuner ==={RESET}\n")

    feasibility_badge = (
        f"{GREEN}{BOLD}✓ FEASIBLE (FITS IN VRAM){RESET}"
        if res["fits_in_vram"]
        else f"{RED}{BOLD}✗ CRITICAL OOM RISK{RESET}"
    )

    headers = ["Metric / Parameter", "Value", "Description"]
    rows = [
        ["Model Architecture", res["model_key"], f"HuggingFace ID: {res['hf_id']}"],
        [
            "GPU SKU & Slices",
            f"{res['num_gpus']}x {res['gpu_sku']}",
            f"Total VRAM: {res['total_gpu_capacity_gib']:.1f} GiB ({res['gpu_capacity_gib']:.1f} GiB/GPU)",
        ],
        [
            "Weights Precision",
            res["dtype"].upper(),
            f"Weights: {res['weight_memory_gib']:.2f} GiB total ({res['weight_memory_per_gpu_gib']:.2f} GiB/GPU)",
        ],
        [
            "KV Cache Precision",
            res["kv_dtype"].upper(),
            f"Per-token KV: {res['per_token_kv_bytes']:,} bytes (GQA-aware)",
        ],
        [
            "CUDA Overhead",
            f"{res['total_cuda_overhead_gib']:.1f} GiB",
            f"Context + Graph Capture ({res['cuda_overhead_gib']:.1f} GiB/GPU)",
        ],
        [
            "Available KV Cache",
            f"{res['available_kv_cache_gib']:.2f} GiB",
            f"Max total KV tokens: {res['max_total_kv_tokens']:,}",
        ],
        [
            "Target Context Window",
            f"{res['target_context']:,} tokens",
            f"Max concurrency at context: {res['max_concurrency_at_target_context']} seqs (target: {res['target_concurrency']})",
        ],
        [
            "Recommended TP Size",
            f"TP = {res['recommended_tp_size']}",
            f"Optimal Tensor Parallelism for NVLink / PCIe",
        ],
        [
            "Recommended --max-model-len",
            f"{res['recommended_max_model_len']:,}",
            "Prevents uncontrolled KV-cache pre-allocation thrashing",
        ],
        [
            "Recommended --gpu-memory-utilization",
            f"{res['gpu_memory_utilization']:.2f}",
            "Headroom buffer for PyTorch CUDA graph and fragmentation",
        ],
        [
            "VRAM Sizing Verdict",
            feasibility_badge,
            "Evaluates weight + CUDA graph + KV cache budget",
        ],
    ]

    print(render_table(headers, rows, title="vLLM Workload Resource Budget"))
    print()

    if res.get("warnings"):
        warn_lines = [f"• {w}" for w in res["warnings"]]
        print(render_panel("Configuration Warnings & Guardrails", warn_lines, border_color=YELLOW))
        print()

    # Recommended vLLM Arguments Panel
    cli_lines = ["vllm serve \\"]
    for arg in res["vllm_cli_args"]:
        cli_lines.append(f"  {arg} \\")
    # Strip trailing backslash from last line
    cli_lines[-1] = cli_lines[-1].rstrip(" \\")

    print(render_panel("Optimized vLLM Runtime Arguments", cli_lines, border_color=GREEN))
    print()

    if args.emit_manifest:
        print(f"{BOLD}{CYAN}# Generated Kubernetes Deployment Manifest:{RESET}")
        print(res["k8s_manifest_yaml"])
        print()

    return 0


def cmd_partition(args: argparse.Namespace) -> int:
    """Recommends GPU partitioning strategy (MIG, MPS, Time-Slicing, or Full GPU)."""
    try:
        res = recommend_partitioning(
            gpu_sku=args.gpu,
            workload_type=args.workload,
            tenancy=args.tenants,
            model_vram_gb=args.model_vram,
        )
    except Exception as e:
        print(f"{RED}Error calculating partitioning advice: {e}{RESET}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(res, indent=2))
        return 0

    print(f"\n{BOLD}{CYAN}=== KubeVRAM GPU Partitioning & Multi-Tenancy Advisor ==={RESET}\n")

    headers = ["Evaluation Factor", "Value"]
    rows = [
        ["Target GPU SKU", res["gpu_sku"]],
        ["Silicon MIG Support", "Yes (Ampere/Hopper/Blackwell)" if res["supports_mig"] else "No (Requires MPS or Time-Slicing)"],
        ["Workload Classification", res["workload_type"]],
        ["Multi-Tenancy Model", res["tenancy"]],
        ["Model VRAM Footprint", f"{res['model_vram_gb']:.1f} GiB"],
        ["Recommended Strategy", f"{BOLD}{res['recommended_strategy']}{RESET}"],
        ["Selected MIG Profile", res.get("mig_profile") or "None (N/A)"],
        ["Implementation Risk", severity_badge(res["risk_level"])],
    ]

    print(render_table(headers, rows, title="GPU Partitioning Recommendation"))
    print()

    # Architectural Rationale
    rationale_lines = [res["rationale"]]
    print(render_panel("Architectural Decision & Rationale", rationale_lines, border_color=CYAN))
    print()

    if res.get("warnings"):
        warn_lines = [f"• {w}" for w in res["warnings"]]
        print(render_panel("Partitioning Warnings & Known Pitfalls", warn_lines, border_color=YELLOW))
        print()

    if args.emit_configmap and res.get("configmap_yaml"):
        print(f"{BOLD}{CYAN}# Generated NVIDIA GPU Operator ConfigMap:{RESET}")
        print(res["configmap_yaml"])
        print()

    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    """Translates Linux/CUDA administration mental models into Kubernetes and DCGM equivalents."""
    query_str = " ".join(args.query).strip() if args.query else None
    entries = translate_linux_intent(query_str)

    if args.format == "json":
        print(json.dumps(entries, indent=2))
        return 0

    print(f"\n{BOLD}{CYAN}=== KubeVRAM Linux-to-Kubernetes & DCGM Abstraction Bridge ==={RESET}\n")

    if not entries:
        print(f"{YELLOW}No translations found matching query: '{query_str}'{RESET}\n")
        print(f"Try querying by Linux command (e.g., 'nvidia-smi', 'dmesg', 'shm', 'topo', 'xid').")
        return 0

    if query_str:
        # Detailed panel mode for targeted queries
        for entry in entries:
            title = f"{entry['category']} | {entry['linux_cmd']}"
            lines = [
                f"{BOLD}Linux Command:{RESET} {entry['linux_cmd']}",
                f"{BOLD}Linux Intent:{RESET} {entry['linux_intent']}",
                "",
                f"{BOLD}Kubernetes Equivalent:{RESET}",
                f"  $ {entry['k8s_equivalent']}",
                "",
                f"{BOLD}DCGM PromQL Metric:{RESET}",
                f"  {entry['dcgm_promql']}",
                "",
                f"{BOLD}KubeVRAM Fast Command:{RESET}",
                f"  $ {entry['kubevram_cmd']}",
                "",
                f"{BOLD}Engineering Explanation:{RESET}",
                f"  {entry['explanation']}",
            ]
            print(render_panel(title, lines, border_color=CYAN))
            print()
    else:
        # Catalog table mode
        headers = ["Category", "Linux Command", "Kubernetes Equivalent", "DCGM PromQL", "KubeVRAM"]
        rows: List[List[str]] = []
        for entry in entries:
            k8s_trunc = (
                entry["k8s_equivalent"][:38] + "…"
                if len(entry["k8s_equivalent"]) > 38
                else entry["k8s_equivalent"]
            )
            prom_trunc = (
                entry["dcgm_promql"][:32] + "…"
                if len(entry["dcgm_promql"]) > 32
                else entry["dcgm_promql"]
            )
            rows.append(
                [
                    entry["category"],
                    entry["linux_cmd"],
                    k8s_trunc,
                    prom_trunc,
                    entry["kubevram_cmd"],
                ]
            )

        print(render_table(headers, rows, title="Linux Admin to Kubernetes Translation Knowledge Base"))
        print(f"\n{DIM}Tip: Run 'kubevram bridge <query>' (e.g., 'kubevram bridge topo' or 'kubevram bridge xid') for full command explanations.{RESET}\n")

    return 0


def cmd_rbac_audit(args: argparse.Namespace) -> int:
    """Audits cluster inference workloads for security posture, privileged flags, and RBAC issues."""
    nodes, pods = get_mock_cluster_state()
    findings = audit_cluster_rbac(nodes, pods)

    if args.format == "json":
        print(json.dumps(findings, indent=2))
        return 0

    print(f"\n{BOLD}{CYAN}=== KubeVRAM GPU Workload Security & RBAC Auditor ==={RESET}\n")

    if not findings:
        print(f"{GREEN}✓ No security misconfigurations or over-privileged GPU workloads detected.{RESET}\n")
        return 0

    headers = ["Check ID", "Severity", "Resource", "Namespace", "Security Issue Summary"]
    rows: List[List[str]] = []
    for f in findings:
        issue_trunc = f["issue"][:58] + "…" if len(f["issue"]) > 58 else f["issue"]
        ns_str = f.get("namespace") or "-"
        rows.append(
            [
                f["check_id"],
                severity_badge(f["severity"]),
                f["resource"],
                ns_str,
                issue_trunc,
            ]
        )

    print(render_table(headers, rows, title="GPU Security & RBAC Audit Findings"))
    print()

    for idx, f in enumerate(findings, 1):
        panel_color = RED if f["severity"] in ("CRITICAL", "HIGH") else YELLOW
        title = f"Finding #{idx}: {f['check_id']} on {f['resource']}"
        lines = [
            f"{BOLD}Resource:{RESET} {f['resource']}   {BOLD}Severity:{RESET} {severity_badge(f['severity'])}",
            f"{BOLD}Issue:{RESET} {f['issue']}",
            "",
            f"{BOLD}Remediation Guidance:{RESET} {f['remediation']}",
        ]
        if f.get("yaml_fix"):
            lines.append("")
            lines.append(f"{BOLD}Declarative YAML Fix:{RESET}")
            for y_line in f["yaml_fix"].split("\n"):
                lines.append(f"  {y_line}")

        print(render_panel(title, lines, border_color=panel_color))
        print()

    return 0


def cmd_overview(args: Optional[argparse.Namespace] = None) -> int:
    """Displays the master KubeVRAM health banner, cluster telemetry summary, and quickstart."""
    nodes, pods = get_mock_cluster_state()
    findings = triage_cluster(nodes, pods)
    audit_findings = audit_cluster_rbac(nodes, pods)

    total_gpus = sum(len(n.gpus) for n in nodes)
    total_vram_mib = sum(g.total_vram_mib for n in nodes for g in n.gpus)
    used_vram_mib = sum(g.used_vram_mib for n in nodes for g in n.gpus)
    xid_count = sum(1 for n in nodes for g in n.gpus if g.xid_error > 0)
    unhealthy_pods = len(findings)
    security_violations = len(audit_findings)

    banner = f"""{BOLD}{CYAN}
  ╦╔═╦ ╦╔╗ ╔═╗╦  ╦╦═╗╔═╗╔╦╗
  ╠╩╗║ ║╠╩╗║╣ ╚╗╔╝╠╦╝╠═╣║║║
  ╩ ╩╚═╝╚═╝╚═╝ ╚╝ ╩╚═╩ ╩╩ ╩{RESET}
  {BOLD}AI-Assisted Kubernetes Inference Bridge & Day-2 GPU Operations CLI{RESET}
  {DIM}Zero-dependency terminal observability for production AI infrastructure{RESET}
"""
    print(banner)

    summary_headers = ["Cluster Dimension", "Value", "Health State"]
    summary_rows = [
        ["GPU Worker Nodes", f"{len(nodes)} nodes (H100, A100, L40S)", f"{GREEN}[READY]{RESET}"],
        ["Total GPU Devices / Slices", f"{total_gpus} devices", f"{GREEN}[ACTIVE]{RESET}"],
        [
            "Cluster VRAM Pressure",
            f"{used_vram_mib:,} / {total_vram_mib:,} MiB",
            vram_bar(used_vram_mib, total_vram_mib, width=14),
        ],
        [
            "Active Hardware XID Faults",
            f"{xid_count} device(s)",
            f"{RED}{BOLD}[FAULT DETECTED]{RESET}" if xid_count > 0 else f"{GREEN}[HEALTHY]{RESET}",
        ],
        [
            "Unhealthy Inference Pods",
            f"{unhealthy_pods} pod(s) with RCA findings",
            f"{RED}[TRIAGE REQUIRED]{RESET}" if unhealthy_pods > 0 else f"{GREEN}[HEALTHY]{RESET}",
        ],
        [
            "Security & RBAC Warnings",
            f"{security_violations} misconfiguration(s)",
            f"{YELLOW}[AUDIT WARNING]{RESET}" if security_violations > 0 else f"{GREEN}[COMPLIANT]{RESET}",
        ],
    ]

    print(render_table(summary_headers, summary_rows, title="Cluster Health Overview"))
    print()

    quickstart_lines = [
        f"{BOLD}1. Inspect GPU Topology & NVLink Fabric:{RESET}",
        "   $ kubevram topology --fabric",
        "",
        f"{BOLD}2. Correlate Pod Crashes & VRAM OOMs (Exit 137 vs Exit 1):{RESET}",
        "   $ kubevram triage",
        "",
        f"{BOLD}3. Size vLLM Weights & KV-Cache Footprint Deterministically:{RESET}",
        "   $ kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --dtype fp8 --kv-dtype fp8",
        "",
        f"{BOLD}4. Generate Hardware-Aware Partitioning ConfigMaps (MIG vs MPS):{RESET}",
        "   $ kubevram partition --gpu H100-80GB --workload vllm --tenants multi --emit-configmap",
        "",
        f"{BOLD}5. Translate Classic Linux Admin Mental Models to Kubernetes:{RESET}",
        "   $ kubevram bridge 'nvidia-smi topo'   or   $ kubevram bridge xid",
        "",
        f"{BOLD}6. Audit GPU Container Security & Privileged Containers:{RESET}",
        "   $ kubevram rbac-audit",
    ]
    print(render_panel("KubeVRAM Quickstart & Key Commands", quickstart_lines, border_color=CYAN))
    print()

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Builds the master argument parser with subcommands and flags."""
    parser = argparse.ArgumentParser(
        prog="kubevram",
        description="KubeVRAM: AI-Assisted Kubernetes Inference Bridge & Day-2 GPU Operations CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # overview
    subparsers.add_parser("overview", help="Display cluster health banner and summary telemetry")

    # topology
    parser_topo = subparsers.add_parser("topology", help="Inspect GPU node inventory, DCGM bars, and NVLink fabric")
    parser_topo.add_argument(
        "--fabric",
        action="store_true",
        help="Display NUMA policy, NVLink/PCIe interconnect fabric, and Gang Scheduling PodGroup state",
    )
    parser_topo.add_argument("--node", type=str, default=None, help="Filter display to a specific node name")
    parser_topo.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    # triage
    parser_triage = subparsers.add_parser(
        "triage",
        help="Correlate pod crashes (Exit 137 vs CUDA Exit 1 vs Time-Slicing vs XID) with root cause analysis",
    )
    parser_triage.add_argument("--pod", type=str, default=None, help="Filter triage analysis to a specific pod name")
    parser_triage.add_argument(
        "--show-patch",
        dest="show_patch",
        action="store_true",
        default=True,
        help="Show declarative YAML remediation patch (default: True)",
    )
    parser_triage.add_argument(
        "--no-patch",
        dest="show_patch",
        action="store_false",
        help="Hide declarative YAML remediation patch",
    )
    parser_triage.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    # tune-vllm
    parser_tune = subparsers.add_parser(
        "tune-vllm",
        help="Calculate exact model weight footprint, KV cache capacity, and optimal vLLM CLI flags",
    )
    parser_tune.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b",
        help="Model identifier from catalog (e.g., llama-3.1-8b, llama-3.1-70b, qwen2.5-32b)",
    )
    parser_tune.add_argument(
        "--gpu",
        type=str,
        default="H100-80GB",
        help="Target GPU hardware SKU or MIG slice (e.g., H100-80GB, A100-80GB, L40S-48GB)",
    )
    parser_tune.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs or target Tensor Parallelism (TP) size (default: 1)",
    )
    parser_tune.add_argument(
        "--dtype",
        type=str,
        default="fp8",
        help="Model weight precision (fp16, bf16, fp8, int4; default: fp8)",
    )
    parser_tune.add_argument(
        "--kv-dtype",
        type=str,
        default="fp8",
        help="KV cache precision (fp16, bf16, fp8; default: fp8)",
    )
    parser_tune.add_argument(
        "--target-context",
        type=int,
        default=16384,
        help="Target sequence context window length in tokens (default: 16384)",
    )
    parser_tune.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Target concurrent active sequences (default: 32)",
    )
    parser_tune.add_argument(
        "--emit-manifest",
        action="store_true",
        default=False,
        help="Emit ready-to-apply Kubernetes Deployment manifest",
    )
    parser_tune.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    # partition
    parser_part = subparsers.add_parser(
        "partition",
        help="Evaluate workload requirements and recommend MIG vs MPS vs Time-Slicing vs Full GPU",
    )
    parser_part.add_argument(
        "--gpu",
        type=str,
        default="H100-80GB",
        help="Target GPU hardware SKU (default: H100-80GB)",
    )
    parser_part.add_argument(
        "--workload",
        type=str,
        default="vllm",
        help="Workload type (vllm, dev-notebook, embeddings, rag-pipeline; default: vllm)",
    )
    parser_part.add_argument(
        "--tenants",
        type=str,
        default="multi",
        help="Multi-tenancy model (single, multi; default: multi)",
    )
    parser_part.add_argument(
        "--model-vram",
        type=float,
        default=16.0,
        help="Required model VRAM in GiB (default: 16.0)",
    )
    parser_part.add_argument(
        "--emit-configmap",
        action="store_true",
        default=False,
        help="Emit generated NVIDIA GPU Operator ConfigMap YAML",
    )
    parser_part.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    # bridge
    parser_bridge = subparsers.add_parser(
        "bridge",
        help="Translate Linux/CUDA administration commands into Kubernetes API and DCGM PromQL queries",
    )
    parser_bridge.add_argument(
        "query",
        nargs="*",
        default=None,
        help="Linux command or keyword to translate (e.g., 'nvidia-smi', 'topo', 'xid', 'shm')",
    )
    parser_bridge.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    # rbac-audit
    parser_rbac = subparsers.add_parser(
        "rbac-audit",
        help="Audit GPU workloads for privileged containers, missing runtimeClassName, and RBAC issues",
    )
    parser_rbac.add_argument("--format", choices=["table", "json"], default="table", help="Output format (table|json)")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution handler."""
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Default to overview if no subcommand provided
    if not argv or argv[0] in ("-h", "--help") and len(argv) == 1:
        # If user explicitly asked for top-level help, let argparse handle it
        if argv and argv[0] in ("-h", "--help"):
            parser.print_help()
            return 0
        return cmd_overview()

    args = parser.parse_args(argv)

    if not args.subcommand or args.subcommand == "overview":
        return cmd_overview(args)
    elif args.subcommand == "topology":
        return cmd_topology(args)
    elif args.subcommand == "triage":
        return cmd_triage(args)
    elif args.subcommand == "tune-vllm":
        return cmd_tune_vllm(args)
    elif args.subcommand == "partition":
        return cmd_partition(args)
    elif args.subcommand == "bridge":
        return cmd_bridge(args)
    elif args.subcommand == "rbac-audit":
        return cmd_rbac_audit(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
