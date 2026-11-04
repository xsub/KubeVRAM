# KubeVRAM Engineering Roadmap & Actionable TODO

> **Architecture Reference**: [docs/architecture.md](./docs/architecture.md)

---

## Phase 1: Core CLI, Correctness Invariants & Scaling Engine (Completed)

- [x] **Project Structure & Zero-Dependency CLI Packaging (`pyproject.toml`, `bin/kubevram`)**
  - [x] Zero-dependency ANSI terminal renderer + `--format json` across all subcommands (including `overview`) + `--fail-on-severity` CI gates.
  - [x] Multi-node Mock GPU Cluster state (`kubevram/mock_cluster.py`) with owner references, per-container states, and scrape/termination timestamps.
- [x] **GPU Topology & Interconnect Fabric Mapper (`kubevram topology`)**
  - [x] Terminal visualization of GPU SKU, sharing mode (`none`, `mig`, `time-slicing`, `mps`), physical vs. virtualized capacity, and per-GPU `DCGM_FI_DEV_FB_USED` bars.
  - [x] `O(nodes + pods)` indexed `PodGroup` fabric grouping and single-pass `ClusterNode.to_dict()` serialization.
- [x] **Evidence-Aware OOM & Hardware Fault Correlator (`kubevram triage`)**
  - [x] Distinguish **Kubernetes System RAM OOM (`OOMKilled`, Exit 137)** vs. **CUDA VRAM OOM (explicit allocation failure + `DCGM_FI_DEV_FB_USED`)** vs. **Time-Slicing VRAM Collision** vs. **NCCL `/dev/shm` `SIGBUS`** vs. **parsed NVIDIA `XID` faults** (silicon/ECC isolation vs. MMU/driver faults).
  - [x] Detect active GPU XIDs on `Running` pods while skipping clean `Succeeded`/`Exit 0` completions and recovered historical restarts.
  - [x] Calibrate confidence from timestamp freshness (`evidence_gaps`) and emit owner/container-aware `kubectl` commands.
- [x] **Static vLLM & KV-Cache Capacity Sizer (`kubevram tune-vllm`)**
  - [x] Exact decimal-parameter-to-binary-GiB conversion, strict dtype validation, and GQA + TP KV-head replication modeling (`max(1, ceil(num_kv_heads / tp))`).
  - [x] Unified sizing and emitted configuration (`--dtype` / `--quantization` / `--kv-cache-dtype` validated against `vllm/vllm-openai:v0.6.4`), GPU-SKU `nodeAffinity`, and multinode `LeaderWorkerSet` manifests for `TP > 8`.
- [x] **Capacity-Validated GPU Partitioning Advisor (`kubevram partition`)**
  - [x] Enforce physical single-GPU capacity, per-family MIG profile tables (`A30`, `A100-40GB`, `A100/H100-80GB`, `H100-NVL`, `H200`, `B200`), and equal per-replica CUDA MPS memory division.
- [x] **Linux-to-K8s Abstraction Bridge (`kubevram bridge`) & Security Auditor (`kubevram rbac-audit`)**
  - [x] Verified executable `kubevram_cmd` shortcuts across all 12 bridge translations.
  - [x] Wildcard secret checks across all ServiceAccounts, CDI/default-runtime-aware `runtimeClassName` checks, and numeric GPU count parsing (`>= 2` GPUs including odd counts).

---

## Phase 2: Live Cluster & Telemetry Collectors (Completed Baseline, Streaming Next)

- [x] **Zero-Dependency Live & Snapshot Collector (`kubevram/collectors.py`)**
  - [x] `--live` collection via `kubectl` (`nodes`, `pods`, `replicasets`, `serviceaccounts`, `roles`, `clusterroles`, `rolebindings`, `clusterrolebindings`, container logs) with `ReplicaSet -> Deployment` owner unwrapping.
  - [x] Optional Prometheus HTTP API (`--prom-url`) and direct `dcgm-exporter` (`--dcgm-url`) scraping for `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`, `DCGM_FI_DEV_GPU_UTIL`, and `DCGM_FI_DEV_XID_ERRORS`.
  - [x] Offline bundle inspection via `--from-file` (accepting raw `kubectl get ... -o json` lists or serialized KubeVRAM snapshots).
- [ ] **Historical Time-Series Range Correlation**
  - [ ] Query Prometheus `/api/v1/query_range` around exact `terminated.finishedAt` timestamps to reconstruct pre-crash VRAM spikes even after container restart.
- [ ] **vLLM / Triton Runtime Metrics Scraper**
  - [ ] Scrape `vllm:gpu_cache_usage_perc`, `vllm:num_requests_waiting`, `vllm:time_to_first_token_seconds`, and `vllm:time_per_output_token_seconds` to measure empirical latency/throughput alongside static VRAM sizing.

---

## Phase 3: AI-Augmented RCA Agent & Natural Language Intent Bridge

- [ ] **Pluggable LLM Diagnostics Provider (`kubevram/ai/provider.py`)**
  - [ ] Support local OpenAI-compatible endpoints (vLLM / Ollama), Google Gemini API, and Anthropic Claude for synthesizing multi-signal RCA reports on novel crash signatures.
  - [ ] Strict redaction filter stripping environment variables, `HF_TOKEN` secrets, and internal IPs before sending diagnostic bundles to external LLMs.
- [ ] **Automated Hardware Fault Playbooks**
  - [ ] Optional `--remediate` workflow for fatal silicon XID errors (`XID 48`, `XID 79`, `XID 94`, `XID 95`): annotate node with fault metadata, taint/cordon node, and safely evict pods respecting `PodDisruptionBudget`.

---

## Phase 4: Day-2 Continuous Optimization & Admission Guardrails

- [ ] **ValidatingAdmissionWebhook for GPU Workloads**
  - [ ] Block `vLLM` or `TGI` deployments from scheduling onto `time-slicing` nodes unless `--gpu-memory-utilization` is explicitly capped below `1 / replicas`.
  - [ ] Enforce `/dev/shm` memory-backed volume presence (`>= 8Gi`) on multi-GPU (`nvidia.com/gpu >= 2`) pods to prevent NCCL bus errors.
- [ ] **Right-Sizing & FinOps Recommender (`kubevram finops`)**
  - [ ] Analyze 7-day `DCGM_FI_DEV_FB_USED` and `DCGM_FI_PROF_SM_ACTIVE` percentiles across inference services and recommend right-sized `MIG` slices or `FP8` quantization profiles.
