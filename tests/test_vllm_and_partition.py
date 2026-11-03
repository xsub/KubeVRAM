"""
Unit tests for kubevram.vllm_calculator and kubevram.partition_advisor.
Uses only standard library unittest.
"""

import unittest
from kubevram.vllm_calculator import (
    MODEL_CATALOG,
    GPU_CATALOG,
    calculate_vllm_tuning,
)
from kubevram.partition_advisor import (
    recommend_partitioning,
    is_mig_supported,
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
            self.assertIn("hf_id", entry)
            self.assertIn("params_b", entry)
            self.assertIn("num_layers", entry)
            self.assertIn("num_kv_heads", entry)
            self.assertIn("head_dim", entry)
            self.assertIn("default_max_len", entry)

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
        self.assertAlmostEqual(res["weight_memory_gib"], 8.03 * 2.0 * 1.04, places=2)
        self.assertEqual(res["cuda_overhead_gib"], 2.8)
        self.assertEqual(res["gpu_memory_utilization"], 0.92)
        self.assertIn("runtimeClassName: nvidia", res["k8s_manifest_yaml"])
        self.assertIn("/dev/shm", res["k8s_manifest_yaml"])
        self.assertIn("sizeLimit: 16Gi", res["k8s_manifest_yaml"])

    def test_calculate_vllm_tuning_llama_70b_fp8_single_gpu(self):
        res = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=1,
            dtype="fp8",
            kv_dtype="fp8",
        )
        # 70.6 * 1.0 * 1.04 = 73.424 + 2.8 = 76.224 > 73.6 (80 * 0.92)
        self.assertFalse(res["fits_in_vram"])
        self.assertGreaterEqual(res["recommended_tp_size"], 2)
        self.assertTrue(len(res["warnings"]) > 0)

    def test_calculate_vllm_tuning_llama_70b_fp8_multi_gpu(self):
        res = calculate_vllm_tuning(
            model_key="llama-3.1-70b",
            gpu_sku="H100-80GB",
            num_gpus=4,
            dtype="fp8",
            kv_dtype="fp8",
        )
        self.assertTrue(res["fits_in_vram"])
        self.assertGreater(res["max_total_kv_tokens"], 0)
        self.assertGreaterEqual(res["max_concurrency_at_target_context"], 32)


class TestPartitionAdvisor(unittest.TestCase):
    def test_is_mig_supported(self):
        self.assertTrue(is_mig_supported("H100-80GB"))
        self.assertTrue(is_mig_supported("A100-80GB"))
        self.assertTrue(is_mig_supported("A100-40GB"))
        self.assertTrue(is_mig_supported("A30"))
        self.assertTrue(is_mig_supported("H200"))
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
        # Explicit warning against time slicing
        has_ts_warning = any("Time-Slicing" in w for w in rec["warnings"])
        self.assertTrue(has_ts_warning)
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
        has_ts_warning = any("Time-Slicing" in w for w in rec["warnings"])
        self.assertTrue(has_ts_warning)

    def test_dev_notebook(self):
        rec = recommend_partitioning(
            gpu_sku="H100-80GB",
            workload_type="dev-notebook",
            tenancy="single",
            model_vram_gb=4.0,
        )
        self.assertEqual(rec["recommended_strategy"], "Time-Slicing")
        self.assertIn("timeSlicing", rec["configmap_yaml"])

    def test_embeddings_on_a10g(self):
        rec = recommend_partitioning(
            gpu_sku="A10G-24GB",
            workload_type="embeddings",
            tenancy="multi",
            model_vram_gb=4.0,
        )
        self.assertFalse(rec["supports_mig"])
        self.assertEqual(rec["recommended_strategy"], "MPS")
        self.assertIn("mps", rec["configmap_yaml"])


if __name__ == "__main__":
    unittest.main()
