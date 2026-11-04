"""
Unit and invariant tests for kubevram.vllm_calculator and kubevram.partition_advisor.
Uses only standard library unittest.
"""

import math
import unittest

from kubevram.partition_advisor import is_mig_supported, recommend_partitioning
from kubevram.vllm_calculator import (
    BYTES_PER_GIB,
    GPU_CATALOG,
    GPU_SKU_METADATA,
    MODEL_CATALOG,
    PARAMS_PER_BILLION,
    VALID_VLLM_DTYPES,
    VALID_VLLM_KV_DTYPES,
    calculate_vllm_tuning,
)


class TestVLLMCalculator(unittest.TestCase):
    def test_model_catalog_completeness(self):
        expected_models = [
            "llama-3.1-8b",
            "llama-3.1-70b",
            "llama-3.1-405b",
            "mistral-7b",
            "qwen2.5-32b",
            "deepseek-r1-32b",
            "mixtral-8x22b",
        ]
        for m in expected_models:
            self.assertIn(m, MODEL_CATALOG)
            entry = MODEL_CATALOG[m]
            for field in (
                "hf_id",
                "params_b",
                "num_layers",
                "num_attention_heads",
                "num_kv_heads",
                "head_dim",
                "default_max_len",
            ):
                self.assertIn(field, entry)

    def test_gpu_catalog_completeness(self):
        expected_gpus = {
            "H100-80GB": 81920,
            "A100-80GB": 81920,
            "A100-40GB": 40960,
            "L40S-48GB": 49152,
            "A10G-24GB": 24576,
            "MIG-3g.40gb": 40960,
            "MIG-2g.20gb": 20480,
            "MIG-1g.10gb": 10240,
        }
        for sku, mib in expected_gpus.items():
            self.assertIn(sku, GPU_CATALOG)
            self.assertEqual(GPU_CATALOG[sku], mib)

    def test_calculate_vllm_tuning_llama_8b(self):
        res = calculate_vllm_tuning(
            model_key="llama-3.1-8b",
            gpu_sku="H100-80GB",
            num_gpus=1,
            dtype="fp16",
            kv_dtype="fp8",
            target_context=16384,
            target_concurrency=32,
        )
        self.assertTrue(res["fits_in_vram"])
        self.assertEqual(res["recommended_tp_size"], 1)
        expected_gib = (8.03 * PARAMS_PER_BILLION * 2.0) / BYTES_PER_GIB
        self.assertAlmostEqual(res["weight_memory_gib"], expected_gib, places=2)
        self.assertEqual(res["cuda_overhead_gib"], 2.8)
        self.assertEqual(res["gpu_memory_utilization"], 0.92)
        self.assertIn("--dtype=float16", res["vllm_cli_args"])
        self.assertIn("--kv-cache-dtype=fp8", res["vllm_cli_args"])
        self.assertIn("runtimeClassName: nvidia", res["k8s_manifest_yaml"])
        self.assertIn("nvidia.com/gpu.product", res["k8s_manifest_yaml"])
        self.assertIn("NVIDIA-H100-80GB-HBM3", res["k8s_manifest_yaml"])
        self.assertIn("/dev/shm", res["k8s_manifest_yaml"])
        self.assertIn("sizeLimit: 16Gi", res["k8s_manifest_yaml"])

    def test_fp32_llama_70b_requires_263_gib_and_is_infeasible_on_2_h100s(self):
        res = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=2,
            dtype="float32",
            kv_dtype="auto",
        )
        # 70.6e9 * 4 / 1024^3 = 263.006 GiB (> 160 GiB capacity across 2x H100)
        self.assertAlmostEqual(res["weight_memory_gib"], 263.006, places=2)
        self.assertFalse(res["fits_in_vram"])
        self.assertEqual(res["vllm_cli_args"], [])
        self.assertEqual(res["k8s_manifest_yaml"], "")

    def test_unrecognized_or_incompatible_dtypes_raise_value_error(self):
        for bad_dtype in ("fp99", "uint3", "unknown_dtype", ""):
            with self.assertRaises(ValueError):
                calculate_vllm_tuning(model_key="llama-3.1-8b", dtype=bad_dtype)

        for bad_kv in ("int4", "int8", "banana", ""):
            with self.assertRaises(ValueError):
                calculate_vllm_tuning(model_key="llama-3.1-8b", dtype="bfloat16", kv_dtype=bad_kv)

        # vLLM 0.6.4 cannot set 16-bit KV cache when compute dtype is float32
        with self.assertRaises(ValueError):
            calculate_vllm_tuning(model_key="llama-3.1-8b", dtype="float32", kv_dtype="fp16")

        for bad_gpus in (0, -2, True):
            with self.assertRaises(ValueError):
                calculate_vllm_tuning(model_key="llama-3.1-8b", num_gpus=bad_gpus)

    def test_calculate_vllm_tuning_llama_70b_fp8_single_gpu_infeasible_and_no_chimera_manifest(self):
        res = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=1,
            dtype="fp8",
            kv_dtype="fp8",
        )
        self.assertFalse(res["fits_in_vram"])
        self.assertEqual(res["num_gpus"], 1)
        self.assertGreaterEqual(res["recommended_tp_size"], 2)
        self.assertTrue(len(res["warnings"]) > 0)
        # Never emit a mismatched 4-GPU manifest alongside 1-GPU sizing
        self.assertEqual(res["vllm_cli_args"], [])
        self.assertEqual(res["k8s_manifest_yaml"], "")

    def test_default_tuning_matches_emitted_deployment_and_retains_full_concurrency(self):
        res_default = calculate_vllm_tuning()
        self.assertTrue(res_default["fits_in_vram"])
        self.assertEqual(res_default["num_gpus"], res_default["recommended_tp_size"])
        self.assertIn(f"--tensor-parallel-size={res_default['num_gpus']}", res_default["vllm_cli_args"])
        self.assertIn("--max-num-seqs=32", res_default["vllm_cli_args"])
        self.assertIn("--dtype=bfloat16", res_default["vllm_cli_args"])

        res_fp8 = calculate_vllm_tuning(dtype="fp8", kv_dtype="fp8")
        self.assertTrue(res_fp8["fits_in_vram"])
        self.assertEqual(res_fp8["num_gpus"], 4)
        self.assertIn("--tensor-parallel-size=4", res_fp8["vllm_cli_args"])
        self.assertIn("--max-num-seqs=32", res_fp8["vllm_cli_args"])
        self.assertIn("--dtype=auto", res_fp8["vllm_cli_args"])
        self.assertIn("--quantization=fp8", res_fp8["vllm_cli_args"])
        self.assertNotIn("--dtype=fp8", res_fp8["vllm_cli_args"])

    def test_kv_head_replication_at_tp16_halves_tokens_per_gib(self):
        # llama-3.1-70b has 8 KV heads; at TP=16 each GPU gets 1 KV head (2x aggregate replication)
        res_tp8 = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=8,
            dtype="fp8",
            kv_dtype="fp8",
        )
        res_tp16 = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=16,
            dtype="fp8",
            kv_dtype="fp8",
        )
        self.assertEqual(res_tp16["per_token_kv_bytes"], 2 * res_tp8["per_token_kv_bytes"])
        expected_tokens_tp16 = int(
            (res_tp16["available_kv_cache_gib"] * BYTES_PER_GIB) // res_tp16["per_token_kv_bytes"]
        )
        self.assertEqual(res_tp16["max_total_kv_tokens"], expected_tokens_tp16)
        self.assertTrue(any("KV Head Replication" in w for w in res_tp16["warnings"]))
        # TP=16 exceeds 8 GPUs/node -> generates multinode LeaderWorkerSet with 8 GPUs per pod
        self.assertIn("kind: LeaderWorkerSet", res_tp16["k8s_manifest_yaml"])
        self.assertIn('nvidia.com/gpu: "8"', res_tp16["k8s_manifest_yaml"])
        self.assertNotIn('nvidia.com/gpu: "16"', res_tp16["k8s_manifest_yaml"])

    def test_qwen_32b_on_a100_40gb_never_emits_zero_max_model_len(self):
        res = calculate_vllm_tuning(
            model_key="qwen2.5-32b",
            gpu_sku="A100-40GB",
            num_gpus=1,
            dtype="fp8",
            kv_dtype="fp16",
            target_context=16384,
        )
        if res["max_total_kv_tokens"] >= 16:
            self.assertGreaterEqual(res["recommended_max_model_len"], 16)
        self.assertNotIn("--max-model-len=0", res["vllm_cli_args"])

    def test_sweep_all_2016_configurations_invariants(self):
        models = list(MODEL_CATALOG.keys())  # 7
        gpus = list(GPU_CATALOG.keys())  # 8
        tp_counts = [1, 2, 3, 4, 8, 16]  # 6
        dtype_pairs = [
            ("bfloat16", "auto"),
            ("fp16", "fp8"),
            ("fp8", "fp8"),
            ("float32", "auto"),
            ("awq", "fp16"),
            ("gptq", "fp8"),
        ]  # 6 -> 7 * 8 * 6 * 6 = 2,016 configurations

        count = 0
        for m in models:
            for g in gpus:
                for tp in tp_counts:
                    for dt, kv_dt in dtype_pairs:
                        count += 1
                        res = calculate_vllm_tuning(
                            model_key=m,
                            gpu_sku=g,
                            num_gpus=tp,
                            dtype=dt,
                            kv_dtype=kv_dt,
                            target_context=4096,
                            target_concurrency=8,
                        )
                        self.assertEqual(res["num_gpus"], tp)
                        self.assertNotIn("--max-model-len=0", res["vllm_cli_args"])
                        if res["max_total_kv_tokens"] >= 16:
                            self.assertGreaterEqual(res["recommended_max_model_len"], 16)

                        if res["fits_in_vram"]:
                            self.assertGreater(res["recommended_max_model_len"], 0)
                            self.assertGreaterEqual(res["max_total_kv_tokens"], 4096)
                            self.assertIn(f"--tensor-parallel-size={tp}", res["vllm_cli_args"])
                            dtype_args = [
                                a.split("=", 1)[1]
                                for a in res["vllm_cli_args"]
                                if a.startswith("--dtype=")
                            ]
                            kv_args = [
                                a.split("=", 1)[1]
                                for a in res["vllm_cli_args"]
                                if a.startswith("--kv-cache-dtype=")
                            ]
                            self.assertEqual(len(dtype_args), 1)
                            self.assertIn(dtype_args[0], VALID_VLLM_DTYPES)
                            self.assertEqual(len(kv_args), 1)
                            self.assertIn(kv_args[0], VALID_VLLM_KV_DTYPES)
                            for lbl in GPU_SKU_METADATA[g]["gfd_product_labels"][:1]:
                                self.assertIn(lbl, res["k8s_manifest_yaml"])
                        else:
                            self.assertEqual(res["vllm_cli_args"], [])
                            self.assertEqual(res["k8s_manifest_yaml"], "")

        self.assertEqual(count, 2016)


class TestPartitionAdvisor(unittest.TestCase):
    def test_is_mig_supported(self):
        self.assertTrue(is_mig_supported("H100-80GB"))
        self.assertTrue(is_mig_supported("A100-80GB"))
        self.assertTrue(is_mig_supported("A100-40GB"))
        self.assertTrue(is_mig_supported("A30"))
        self.assertTrue(is_mig_supported("H200"))
        self.assertTrue(is_mig_supported("B200"))
        self.assertTrue(is_mig_supported("MIG-3g.40gb"))

        self.assertFalse(is_mig_supported("L40S-48GB"))
        self.assertFalse(is_mig_supported("A10G-24GB"))
        self.assertFalse(is_mig_supported("T4"))
        self.assertFalse(is_mig_supported("L4"))

    def test_vllm_multi_tenant_on_h100(self):
        rec = recommend_partitioning(
            gpu_sku="H100-80GB",
            workload_type="vllm",
            tenancy="multi",
            model_vram_gb=16.0,
        )
        self.assertTrue(rec["supports_mig"])
        self.assertEqual(rec["recommended_strategy"], "MIG")
        self.assertEqual(rec["mig_profile"], "2g.20gb")
        self.assertEqual(rec["risk_level"], "LOW")
        self.assertTrue(any("Time-Slicing" in w for w in rec["warnings"]))
        self.assertIn("custom-mig-parted-configs", rec["configmap_yaml"])

    def test_vllm_multi_tenant_on_l40s(self):
        rec = recommend_partitioning(
            gpu_sku="L40S-48GB",
            workload_type="vllm",
            tenancy="multi",
            model_vram_gb=16.0,
        )
        self.assertFalse(rec["supports_mig"])
        self.assertEqual(rec["recommended_strategy"], "Full-GPU")
        self.assertEqual(rec["risk_level"], "HIGH")

    def test_dev_notebook(self):
        rec = recommend_partitioning(
            gpu_sku="H100-80GB",
            workload_type="dev-notebook",
            tenancy="single",
            model_vram_gb=4.0,
        )
        self.assertEqual(rec["recommended_strategy"], "Time-Slicing")
        self.assertIn("timeSlicing", rec["configmap_yaml"])

    def test_embeddings_4gb_fits_mps_on_a10g_but_20gb_requires_full_gpu(self):
        rec_small = recommend_partitioning(
            gpu_sku="A10G-24GB",
            workload_type="embeddings",
            tenancy="multi",
            model_vram_gb=4.0,
        )
        self.assertEqual(rec_small["recommended_strategy"], "MPS")
        self.assertEqual(rec_small["risk_level"], "LOW")

        # 20 GiB on 24 GB A10G cannot fit inside 2 equal MPS replicas (~12 GB/replica)
        rec_large = recommend_partitioning(
            gpu_sku="A10G-24GB",
            workload_type="embeddings",
            tenancy="multi",
            model_vram_gb=20.0,
        )
        self.assertEqual(rec_large["recommended_strategy"], "Full-GPU")
        self.assertEqual(rec_large["risk_level"], "HIGH")
        self.assertNotIn("mps:", rec_large["configmap_yaml"])

    def test_workload_exceeding_device_capacity_flagged_critical(self):
        rec = recommend_partitioning(
            gpu_sku="A10G-24GB",
            workload_type="embeddings",
            tenancy="single",
            model_vram_gb=30.0,
        )
        self.assertEqual(rec["recommended_strategy"], "Full-GPU")
        self.assertEqual(rec["risk_level"], "CRITICAL")

    def test_gpu_family_specific_mig_profiles(self):
        rec_a30 = recommend_partitioning(
            gpu_sku="A30",
            workload_type="embeddings",
            tenancy="multi",
            model_vram_gb=5.0,
        )
        self.assertEqual(rec_a30["recommended_strategy"], "MIG")
        self.assertEqual(rec_a30["mig_profile"], "1g.6gb")

        rec_a30_large = recommend_partitioning(
            gpu_sku="A30",
            workload_type="embeddings",
            tenancy="multi",
            model_vram_gb=18.0,
        )
        self.assertEqual(rec_a30_large["recommended_strategy"], "Full-GPU")
        self.assertIsNone(rec_a30_large["mig_profile"])

        rec_h200 = recommend_partitioning(
            gpu_sku="H200",
            workload_type="vllm",
            tenancy="multi",
            model_vram_gb=25.0,
        )
        self.assertEqual(rec_h200["recommended_strategy"], "MIG")
        self.assertEqual(rec_h200["mig_profile"], "2g.35gb")

        rec_a100_40 = recommend_partitioning(
            gpu_sku="A100-40GB",
            workload_type="triton",
            tenancy="multi",
            model_vram_gb=30.0,
        )
        self.assertEqual(rec_a100_40["recommended_strategy"], "Full-GPU")
        self.assertIsNone(rec_a100_40["mig_profile"])

    def test_invalid_vram_and_tenancy_inputs_rejected(self):
        for bad_vram in (-5.0, 0.0, math.nan, math.inf, -math.inf, True, "not-a-number"):
            with self.assertRaises(ValueError):
                recommend_partitioning(gpu_sku="H100-80GB", model_vram_gb=bad_vram)

        with self.assertRaises(ValueError):
            recommend_partitioning(gpu_sku="H100-80GB", tenancy="cluster-wide", model_vram_gb=8.0)

        with self.assertRaises(ValueError):
            recommend_partitioning(gpu_sku="UNKNOWN-GPU", model_vram_gb=8.0)


if __name__ == "__main__":
    unittest.main()
