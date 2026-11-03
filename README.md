# KubeVRAM

**Terminal-centric Kubernetes GPU & LLM inference operations CLI: Linux-to-K8s bridge, CUDA VRAM vs. K8s OOM triage, vLLM KV-cache tuner, and MIG/MPS advisor.**

`KubeVRAM` bridges traditional Linux/CUDA system administration mental models (`nvidia-smi`, `nvidia-smi topo -m`, `dmesg`, `ipcs -m`, `numactl`) directly to Kubernetes, NVIDIA DCGM hardware telemetry, and modern AI inference runtimes (`vLLM`, `Triton`, `KServe`).

---

## Key Capabilities

- **GPU Topology & Fabric Mapper (`kubevram topology`)**: Visualize node GPUs, hardware `MIG` slices, `Time-Slicing` over-subscription, real-time `DCGM_FI_DEV_FB_USED` saturation gauges, SM occupancy, NUMA alignment, and NVLink/NVSwitch vs. PCIe interconnects.
- **Automated OOM & Hardware Fault Triage (`kubevram triage`)**: Deterministically separate **Kubernetes System RAM `OOMKilled` (`Exit 137`)** from **CUDA VRAM OOM (`Exit 1`)**, **Time-Slicing co-tenant VRAM collisions**, **NCCL `/dev/shm` `SIGBUS` (`Exit 135`)**, and **NVIDIA Hardware `XID` faults**—complete with `kubectl` commands and declarative YAML patches.
- **Deterministic vLLM & KV-Cache Tuner (`kubevram tune-vllm`)**: Compute exact weight footprints, CUDA graph overhead, GQA per-token KV cache bytes, concurrency limits, and optimal `--gpu-memory-utilization` / `--max-model-len` flags, plus ready-to-apply Kubernetes manifests.
- **GPU Partitioning Advisor (`kubevram partition`)**: Hardware-aware decision engine choosing between **MIG**, **CUDA MPS**, **Time-Slicing**, and **Full-GPU** with generated Device Plugin / `mig-parted` ConfigMaps.
- **Linux-to-Kubernetes Abstraction Bridge (`kubevram bridge`)**: Translate classic Linux/CUDA commands into exact `kubectl` queries and DCGM PromQL expressions.
- **GPU Security & RBAC Auditor (`kubevram rbac-audit`)**: Detect unnecessary `privileged: true` GPU containers, missing `runtimeClassName: nvidia`, wildcard secret access, undersized `/dev/shm`, and missing Gang Scheduling `PodGroups`.

---

## Quickstart

`KubeVRAM` runs out-of-the-box on **Python 3.10+** with **zero required external dependencies** (includes a built-in multi-node H100 / A100-MIG / L40S simulation cluster for instant offline exploration and sizing).

```bash
git clone https://github.com/xsub/KubeVRAM.git
cd KubeVRAM

# Optional: editable pip install
pip install -e .

# Or run directly via ./bin/kubevram or python3 -m kubevram
./bin/kubevram
```

### Example Commands

```bash
# 1. Inspect GPU topology, DCGM VRAM bars, and NVSwitch/NVLink/NUMA fabric
./bin/kubevram topology --fabric

# 2. Triage crashing inference pods (Exit 137 vs CUDA VRAM OOM vs Time-Slicing vs XID)
./bin/kubevram triage
./bin/kubevram triage --pod vllm-mistral-7b-tenant-b

# 3. Calculate vLLM KV-cache budget & emit a tuned Kubernetes Deployment manifest
./bin/kubevram tune-vllm --model llama-3.1-70b --gpu H100-80GB --num-gpus 2 --dtype fp8 --kv-dtype fp8 --emit-manifest

# 4. Recommend GPU sharing strategy (MIG vs MPS vs Time-Slicing) & generate ConfigMap
./bin/kubevram partition --gpu H100-80GB --workload vllm --tenants multi --model-vram 16 --emit-configmap

# 5. Translate Linux sysadmin intents to Kubernetes API & DCGM PromQL
./bin/kubevram bridge "nvidia-smi topo"
./bin/kubevram bridge xid

# 6. Audit GPU container privileges, RuntimeClass, /dev/shm, and RBAC posture
./bin/kubevram rbac-audit
```

All commands support `--format json` for CI/CD and automation pipelines.

---

## Repository Layout

```text
KubeVRAM/
├── bin/kubevram                 # Zero-dependency executable entrypoint
├── docs/architecture.md         # Architecture, design spec & telemetry correlation model
├── kubevram/
│   ├── cli.py                   # CLI parser & command handlers
│   ├── ui.py                    # ANSI Unicode tables, gauges & callout panels
│   ├── models.py                # Node, GPU, Pod & RCA domain models
│   ├── oom_correlator.py        # K8s Exit 137 vs CUDA OOM vs XID correlation engine
│   ├── vllm_calculator.py       # GQA KV-cache & VRAM sizing calculator
│   ├── partition_advisor.py     # MIG / MPS / Time-Slicing advisor & YAML generator
│   ├── linux_bridge.py          # Linux-to-K8s & DCGM PromQL translator
│   ├── rbac_auditor.py          # GPU security, RuntimeClass & RBAC auditor
│   └── mock_cluster.py          # H100 / A100-MIG / L40S-Time-Sliced simulation cluster
├── tests/                       # Unit & CLI integration test suite
├── TODO.md                      # Phased engineering roadmap (Phases 1–4)
└── pyproject.toml               # Python package configuration
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
