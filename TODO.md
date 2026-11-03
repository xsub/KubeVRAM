# KubeVRAM Engineering Roadmap & Actionable TODO

> **Architecture Reference**: [docs/architecture.md](./docs/architecture.md)

---

## Phase 1: Core CLI MVP & Offline Simulation Engine (Completed)

- [x] **Project Structure & Zero-Dependency CLI Packaging (`pyproject.toml`, `bin/kubevram`)**
  - [x] Zero-dependency ANSI terminal renderer (Unicode box-drawing tables, VRAM gauge bars, severity badges) + `--format json` output mode.
  - [x] Multi-node Mock GPU Cluster state (`kubevram/mock_cluster.py`) modeling H100 SXM5 (NVSwitch), A100 80GB (MIG `mixed`), and L40S 48GB (Time-Slicing) nodes.
- [x] **GPU Topology & Interconnect Fabric Mapper (`kubevram topology`)**
  - [x] Terminal visualization of GPU SKU, sharing mode (`none`, `mig`, `time-slicing`, `mps`), physical vs. virtualized capacity, and per-GPU `DCGM_FI_DEV_FB_USED` bars.
  - [x] `--fabric` flag displaying NUMA alignment, NVLink/NVSwitch vs. PCIe interconnect topology, and Gang Scheduling (`PodGroup`) status.
- [x] **OOM & Hardware Fault Correlator (`kubevram triage`)**
  - [x] Distinguish **Kubernetes System RAM OOM (`OOMKilled`, Exit Code 137)** vs. **CUDA VRAM OOM (Exit Code 1 + `DCGM_FI_DEV_FB_USED` saturation)** vs. **Time-Slicing VRAM Collision** vs. **NCCL `/dev/shm` Exhaustion (Exit Code 135)** vs. **NVIDIA Hardware XID Faults (`XID 95`)**.
  - [x] Generate context-aware Root Cause Analysis (RCA) with copy-pasteable `kubectl` commands and declarative YAML remediation patches.
- [x] **Deterministic vLLM & KV-Cache Tuner (`kubevram tune-vllm`)**
  - [x] Built-in model catalog (`Llama-3.1-8B`, `Llama-3.1-70B`, `Llama-3.1-405B`, `Mistral-7B`, `Qwen2.5-32B`, `DeepSeek-R1-Distill-32B`, `Mixtral-8x22B`).
  - [x] Calculate exact weight footprint, CUDA graph overhead, per-token KV cache bytes (GQA-aware), max concurrent sequences, and optimal `--gpu-memory-utilization` / `--max-model-len` flags.
  - [x] Emit ready-to-apply Kubernetes `Deployment` YAML (`--emit-manifest`).
- [x] **GPU Partitioning Advisor & Manifest Generator (`kubevram partition`)**
  - [x] Decision engine recommending `MIG`, `Time-Slicing`, `MPS`, or `Full-GPU` based on GPU SKU, workload type (`vllm`, `triton`, `embeddings`, `rag-pipeline`, `dev-notebook`), and multi-tenancy SLA.
  - [x] Generate `mig-parted` ConfigMap, `time-slicing` Device Plugin ConfigMap, or `mps` sharing configuration YAML (`--emit-configmap`).
- [x] **Linux-to-K8s Abstraction Bridge (`kubevram bridge`)**
  - [x] Translate classic Linux/GPU sysadmin commands (`nvidia-smi`, `nvidia-smi topo -m`, `dmesg`, `free -h`, `ipcs -m`, `numactl`, `ulimit -l`) into Kubernetes API calls, DCGM PromQL queries, and KubeVRAM commands.
- [x] **Automated GPU RBAC & Security Auditor (`kubevram rbac-audit`)**
  - [x] Detect unnecessarily privileged inference containers (`privileged: true` instead of `runtimeClassName: nvidia`), wildcard secret access, `/dev/shm` sizing issues, and missing Gang Scheduling `PodGroups`.

---

## Phase 2: Live Cluster & Telemetry Collectors (Next Milestone)

- [ ] **Kubernetes API Live Client (`kubevram/collectors/k8s_client.py`)**
  - [ ] Load kubeconfig (`~/.kube/config` or in-cluster ServiceAccount token) to query live `Nodes`, `Pods`, `Events`, `RuntimeClasses`, and `PodGroups` (`scheduling.volcano.sh/v1beta1`).
  - [ ] Parse `nvidia.com/gpu.*` and `nvidia.com/mig-*` node labels populated by GPU Feature Discovery (GFD).
  - [ ] Stream tail of terminated container logs (`previous=True`) for crashed inference pods.
- [ ] **Prometheus / DCGM Exporter Collector (`kubevram/collectors/dcgm_client.py`)**
  - [ ] Query Prometheus HTTP API (`/api/v1/query`, `/api/v1/query_range`) or scrape node `dcgm-exporter:9400/metrics` directly via port-forward.
  - [ ] Correlate 15-minute trailing time series of `DCGM_FI_DEV_FB_USED`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, and `DCGM_FI_DEV_XID_ERRORS` at the exact timestamp of pod termination.
- [ ] **vLLM / Triton Runtime Metrics Scraper**
  - [ ] Scrape `vllm:gpu_cache_usage_perc`, `vllm:num_requests_waiting`, `vllm:time_to_first_token_seconds`, and `vllm:time_per_output_token_seconds` to detect KV-cache preemption storms before crashes occur.

---

## Phase 3: AI-Augmented RCA Agent & Natural Language Intent Bridge

- [ ] **Pluggable LLM Diagnostics Provider (`kubevram/ai/provider.py`)**
  - [ ] Support local OpenAI-compatible endpoints (vLLM / Ollama), Google Gemini API, and Anthropic Claude for synthesizing multi-signal RCA reports on novel crash signatures.
  - [ ] Strict redaction filter stripping environment variables, `HF_TOKEN` secrets, and internal IPs before sending diagnostic bundles to external LLMs.
- [ ] **Interactive Natural Language Linux-to-K8s Bridge (`kubevram ask "..."`)**
  - [ ] Translate free-form Linux sysadmin intents (e.g., *"Why is GPU 2 on node-gpu-01 throttling and dropping tokens?"* or *"Pin my 70B model to NVLink GPUs on NUMA 0"*) into validated `kubectl` commands and declarative YAML patches.
- [ ] **Automated Hardware Fault Playbooks**
  - [ ] Optional `--remediate` workflow for fatal XID errors (`XID 48`, `XID 79`, `XID 95`): annotate node with fault metadata, taint/cordon node (`nvidia.com/gpu-xid-fault:NoSchedule`), and safely evict pods respecting `PodDisruptionBudget`.

---

## Phase 4: Day-2 Continuous Optimization & Admission Guardrails

- [ ] **ValidatingAdmissionWebhook for GPU Workloads**
  - [ ] Block `vLLM` or `TGI` deployments from scheduling onto `time-slicing` nodes unless `--gpu-memory-utilization` is explicitly capped below `1 / replicas`.
  - [ ] Enforce `/dev/shm` memory-backed volume presence (`>= 8Gi`) on multi-GPU (`nvidia.com/gpu >= 2`) pods to prevent NCCL bus errors.
  - [ ] Enforce `runtimeClassName: nvidia` and reject `securityContext.privileged: true` on non-operator namespaces.
- [ ] **Right-Sizing & FinOps Recommender (`kubevram finops`)**
  - [ ] Analyze 7-day `DCGM_FI_DEV_FB_USED` and `DCGM_FI_PROF_SM_ACTIVE` percentiles across all inference services and recommend right-sized `MIG` slices or `FP8` quantization profiles.
