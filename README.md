# KubeVRAM

**Terminal-centric Kubernetes GPU & LLM inference operations CLI: Linux-to-K8s bridge, evidence-aware CUDA VRAM vs. K8s OOM triage, static vLLM KV-cache sizer, and MIG/MPS partitioning advisor.**

`KubeVRAM` bridges traditional Linux/CUDA system administration mental models (`nvidia-smi`, `nvidia-smi topo -m`, `dmesg`, `ipcs -m`, `numactl`) directly to Kubernetes, NVIDIA DCGM hardware telemetry, and modern AI inference runtimes (`vLLM`, `Triton`, `KServe`).

---

## Key Capabilities

- **Cluster Overview & CI Severity Gates (`kubevram overview`)**: Cluster-wide VRAM utilization, active XID count, unhealthy pod count, and RBAC summary in terminal or `--format json` mode, with `--fail-on-severity` exit-code gating (`exit 2`).
- **GPU Topology & Fabric Mapper (`kubevram topology`)**: Visualize node GPUs, hardware `MIG` slices, `Time-Slicing` over-subscription, real-time `DCGM_FI_DEV_FB_USED` saturation gauges, SM occupancy, NUMA alignment, and NVLink/NVSwitch vs. PCIe interconnects (`O(nodes + pods)` indexed).
- **Evidence-Aware OOM & Hardware Fault Triage (`kubevram triage`)**: Separate **Kubernetes System RAM `OOMKilled` (`Exit 137`)** from **CUDA VRAM OOM (`Exit 1`)**, **Time-Slicing co-tenant VRAM collisions**, **NCCL `/dev/shm` `SIGBUS` (`Exit 135`)**, and **NVIDIA `XID` faults** (silicon/ECC isolation vs. MMU/application faults)—with timestamp freshness checks, calibrated confidence, and owner/container-aware `kubectl` commands.
- **Static vLLM & KV-Cache Capacity Sizer (`kubevram tune-vllm`)**: Compute exact binary-GiB weight footprints, CUDA graph overhead, GQA + TP KV-head-replication token budgets (`max(1, ceil(num_kv_heads / tp))`), and validated `vllm/vllm-openai:v0.6.4` CLI flags, plus GPU-SKU-targeted Kubernetes `Deployment` (`TP <= 8`) or `LeaderWorkerSet` (`TP > 8`) manifests.
- **Capacity-Validated GPU Partitioning Advisor (`kubevram partition`)**: Hardware-aware decision engine choosing between family-specific **MIG** profiles (`A30`, `A100-40GB`, `A100/H100-80GB`, `H100-NVL`, `H200`, `B200`), **CUDA MPS** (accounting for equal per-replica device memory limits), **Time-Slicing**, and **Full-GPU** with generated Device Plugin / `mig-parted` ConfigMaps.
- **Linux-to-Kubernetes Abstraction Bridge (`kubevram bridge`)**: Translate classic Linux/CUDA commands into exact `kubectl` queries, DCGM PromQL expressions, and executable `kubevram` shortcuts.
- **GPU Security & RBAC Auditor (`kubevram rbac-audit`)**: Detect unnecessary `privileged: true` GPU containers, CDI/default-runtime-aware `runtimeClassName` misconfigurations, wildcard secret access across all `ServiceAccounts`, undersized `/dev/shm`, and missing Gang Scheduling `PodGroups` for arbitrary multi-GPU counts.

---

## Quickstart

`KubeVRAM` runs out-of-the-box on **Python 3.10+** with **zero required external Python dependencies**.

```bash
git clone https://github.com/xsub/KubeVRAM.git
cd KubeVRAM

# Optional: editable pip install
pip install -e .

# Or run directly via ./bin/kubevram or python3 -m kubevram
./bin/kubevram
```

---

## Data Collection Modes

Cluster-facing commands (`overview`, `topology`, `triage`, `rbac-audit`) support three explicit input sources:

1. **Live Cluster (`--live`)**: Queries `Nodes`, `Pods`, `ReplicaSets` (`ReplicaSet -> Deployment` owner unwrapping), `ServiceAccounts`, RBAC `Roles`/`ClusterRoles`/`Bindings`, and container logs via `kubectl`, with optional Prometheus (`--prom-url`) or direct `dcgm-exporter` (`--dcgm-url`) scraping for `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_GPU_UTIL`, and `DCGM_FI_DEV_XID_ERRORS`. Fails explicitly on collection errors rather than falling back to mock fixtures.
2. **Captured Snapshot (`--from-file <path.json>`)**: Offline/post-mortem analysis of captured `kubectl get nodes,pods,rs -o json` lists or serialized `KubeVRAM` JSON bundles.
3. **Built-in Simulation (default)**: Multi-node `H100 SXM5` (NVSwitch), `A100 80GB` (`mixed` MIG), and `L40S 48GB` (`time-slicing`) fixture cluster for offline exploration and testing.

---

## CLI Reference & Examples

| Subcommand | Purpose | Key Flags |
| :--- | :--- | :--- |
| `overview` | Cluster GPU health banner, VRAM utilization & triage/RBAC counts | `--live`, `--from-file`, `--format {table,json}`, `--fail-on-severity` |
| `topology` | Node GPU inventory, MIG/Time-Slicing layout & DCGM VRAM bars | `--fabric`, `--node`, `--live`, `--from-file`, `--format {table,json}` |
| `triage` | 5-way root-cause correlation (Exit 137 vs CUDA OOM vs `/dev/shm` vs XID) | `--pod`, `--no-patch`, `--live`, `--from-file`, `--fail-on-severity` |
| `tune-vllm` | Static weight + KV-cache capacity sizing & K8s manifest generation | `--model`, `--gpu`, `--num-gpus`, `--dtype`, `--kv-dtype`, `--emit-manifest` |
| `partition` | MIG vs. CUDA MPS vs. Time-Slicing vs. Full-GPU advisor | `--gpu`, `--workload`, `--tenants`, `--model-vram`, `--emit-configmap` |
| `bridge` | Translate Linux/CUDA sysadmin commands to `kubectl` & DCGM PromQL | `<query>`, `--format {table,json}` |
| `rbac-audit` | Audit `privileged` GPU pods, `RuntimeClass`/CDI, `/dev/shm` & RBAC | `--live`, `--from-file`, `--format {table,json}`, `--fail-on-severity` |

### Example Workflows

```bash
# 1. Cluster overview (terminal dashboard or machine-readable JSON + CI severity gate)
./bin/kubevram overview
./bin/kubevram overview --format json --fail-on-severity critical

# 2. Inspect GPU topology, DCGM VRAM bars, and NVSwitch/NVLink/NUMA fabric
./bin/kubevram topology --fabric

# 3. Triage crashing or XID-degraded inference pods (simulation, live cluster, or snapshot file)
./bin/kubevram triage
./bin/kubevram triage --live --namespace ai-inference --prom-url http://localhost:9090
./bin/kubevram triage --from-file cluster-snapshot.json --fail-on-severity high

# 4. Calculate static vLLM KV-cache budget & emit hardware-targeted Kubernetes manifests
# Auto-selects minimum fitting TP so sizing metrics and emitted manifests always match:
./bin/kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --emit-manifest
./bin/kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --num-gpus 2 --dtype fp8 --kv-dtype fp8 --emit-manifest
# Multinode TP=16 emits a LeaderWorkerSet + Ray manifest across 2x 8-GPU pods:
./bin/kubevram tune-vllm --model llama-3.1-405b --gpu H100-80GB --num-gpus 16 --dtype fp8 --kv-dtype fp8 --emit-manifest

# 5. Recommend GPU sharing strategy & generate GPU Operator / mig-parted ConfigMap
./bin/kubevram partition --gpu H100-80GB --workload vllm --tenants multi --model-vram 16 --emit-configmap

# 6. Translate Linux sysadmin intents to Kubernetes API & DCGM PromQL
./bin/kubevram bridge "nvidia-smi topo"
./bin/kubevram bridge xid

# 7. Audit GPU container privileges, RuntimeClass/CDI, /dev/shm, and RBAC posture
./bin/kubevram rbac-audit --fail-on-severity high
```

### Exit Codes for CI/CD Automation

- `0` — Command succeeded (and no findings met or exceeded `--fail-on-severity`).
- `1` — Invalid arguments, live/file collection failure, or `--emit-manifest` requested on an infeasible configuration (`fits_in_vram: false`).
- `2` — One or more findings met or exceeded `--fail-on-severity {low,medium,high,critical}`.

---

## Supported Catalogs & Scope Guarantees

- **Built-in Model Catalog**: `llama-3.1-8b`, `llama-3.1-70b`, `llama-3.1-405b`, `mistral-7b`, `qwen2.5-32b`, `deepseek-r1-32b`, `mixtral-8x22b`.
- **Validated Weight & KV Precisions**: `--dtype` supports `bfloat16` (default), `float16`, `float32`, `fp8`, `int8`, `awq`, `gptq`, `int4`; `--kv-dtype` supports `auto` (default), `fp8`, `fp8_e4m3`, `fp8_e5m2`, `bfloat16`, `float16`, `float32`. All emitted flags conform to `vllm/vllm-openai:v0.6.4` (`--dtype` + `--quantization`).
- **GPU SKUs & Slices**: `H100-80GB`, `A100-80GB`, `A100-40GB`, `L40S-48GB`, `A10G-24GB`, `MIG-3g.40gb`, `MIG-2g.20gb`, `MIG-1g.10gb` (plus family-specific MIG tables for `A30`, `H100-NVL`, `H200`, and `B200` in `partition`).
- **Static Sizing Scope**: `tune-vllm` establishes **static VRAM and KV-cache capacity feasibility** (including decimal-to-GiB conversion and TP KV-head replication). Runtime latency (TTFT/TPOT) and throughput optimality depend on prompt/output distributions, interconnect bandwidth, and kernel scheduling, and should be validated with live benchmarks.

---

## Repository Layout

```text
KubeVRAM/
├── bin/kubevram                            # Zero-dependency executable entrypoint
├── docs/architecture.md                    # Architecture, evidence model & sizing scope spec
├── kubevram/
│   ├── cli.py                              # CLI parser, live/snapshot/mock routing & severity gates
│   ├── collectors.py                       # Zero-dependency live kubectl + Prometheus/DCGM & file collector
│   ├── ui.py                               # ANSI Unicode tables, gauges & callout panels
│   ├── models.py                           # Node, GPU, ContainerState, Pod & RCA domain models
│   ├── oom_correlator.py                   # Indexed K8s Exit 137 vs CUDA OOM vs XID correlation engine
│   ├── vllm_calculator.py                  # Static GQA/TP KV-cache sizer & Deployment/LWS generator
│   ├── partition_advisor.py                # Capacity- & family-aware MIG / MPS / Time-Slicing advisor
│   ├── linux_bridge.py                     # Linux-to-K8s & DCGM PromQL translator
│   ├── rbac_auditor.py                     # GPU security, RuntimeClass/CDI & RBAC auditor
│   └── mock_cluster.py                     # H100 / A100-MIG / L40S-Time-Sliced simulation cluster
├── tests/
│   ├── test_vllm_and_partition.py          # Sizing invariants, 2,016-config sweep & MIG/MPS tests
│   ├── test_triage_rbac_and_collectors.py  # Triage evidence, RBAC, collector & 24k-pod scaling tests
│   └── test_ui_and_cli.py                  # Terminal renderer, CLI subcommands & severity gate tests
├── TODO.md                                 # Phased engineering roadmap
└── pyproject.toml                          # Python package configuration
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

---

## Documentation & Roadmap

- [Architecture & Specification](./docs/architecture.md)
- [Engineering Roadmap & TODO](./TODO.md)
- [Companion Engineering Handbook: AiOps on K8S — Foundations and Practices](https://github.com/xsub/AiOps_on_K8S_Fundamentals_n_Practicies)

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
