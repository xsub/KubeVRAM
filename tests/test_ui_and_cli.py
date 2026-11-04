"""
Unit and integration tests for KubeVRAM UI rendering, CLI automation contract,
bridge shortcut validity, JSON export, and severity gating.
"""

import io
import json
import shlex
import unittest
from unittest.mock import patch

from kubevram import ui
from kubevram.cli import main
from kubevram.collectors import CollectorError
from kubevram.linux_bridge import BRIDGE_KNOWLEDGE_BASE


class TestUI(unittest.TestCase):
    def test_strip_ansi(self):
        text = f"{ui.RED}{ui.BOLD}Critical Error{ui.RESET}"
        self.assertEqual(ui.strip_ansi(text), "Critical Error")

    def test_vram_bar_thresholds(self):
        bar_low = ui.vram_bar(5000, 10000, width=10)
        self.assertIn("50.0%", bar_low)
        self.assertIn("█████░░░░░", bar_low)
        self.assertIn(ui.GREEN, bar_low)

        bar_mid = ui.vram_bar(8000, 10000, width=10)
        self.assertIn("80.0%", bar_mid)
        self.assertIn(ui.YELLOW, bar_mid)

        bar_high = ui.vram_bar(9500, 10000, width=10)
        self.assertIn("95.0%", bar_high)
        self.assertIn(ui.RED, bar_high)

    def test_severity_badge(self):
        self.assertIn("CRITICAL", ui.severity_badge("CRITICAL"))
        self.assertIn(ui.RED, ui.severity_badge("CRITICAL"))
        self.assertIn("HIGH", ui.severity_badge("HIGH"))
        self.assertIn(ui.RED, ui.severity_badge("HIGH"))
        self.assertIn("MEDIUM", ui.severity_badge("MEDIUM"))
        self.assertIn(ui.YELLOW, ui.severity_badge("MEDIUM"))
        self.assertIn("LOW", ui.severity_badge("LOW"))
        self.assertIn(ui.CYAN, ui.severity_badge("LOW"))
        self.assertIn("OK", ui.severity_badge("OK"))
        self.assertIn(ui.GREEN, ui.severity_badge("OK"))

    def test_render_table_alignment(self):
        headers = ["Col A", "Col B"]
        rows = [
            [f"{ui.RED}Value 1{ui.RESET}", "Short"],
            ["Normal", f"{ui.GREEN}Longer Value 2{ui.RESET}"],
        ]
        table = ui.render_table(headers, rows, title="Test Table")
        lines = table.split("\n")
        self.assertGreater(len(lines), 4)
        line_lengths = [len(ui.strip_ansi(l)) for l in lines]
        self.assertEqual(len(set(line_lengths)), 1, f"Table line lengths differ: {line_lengths}")

    def test_render_panel_alignment(self):
        lines = ["Line 1", "Line 2 is somewhat longer", "Line 3 with \x1b[31mcolor\x1b[0m"]
        panel = ui.render_panel("Panel Title", lines, border_color=ui.CYAN)
        p_lines = panel.split("\n")
        line_lengths = [len(ui.strip_ansi(l)) for l in p_lines]
        self.assertEqual(len(set(line_lengths)), 1, f"Panel line lengths differ: {line_lengths}")


class TestCLI(unittest.TestCase):
    def test_cli_overview_table_and_json(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code = main(["overview"])
            self.assertEqual(code, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["overview", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertEqual(data["node_count"], 3)
            self.assertIn("triage_findings", data)
            self.assertIn("rbac_findings", data)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out2:
            code_top_json = main(["--format", "json"])
            self.assertEqual(code_top_json, 0)
            data2 = json.loads(mock_out2.getvalue())
            self.assertEqual(data2["node_count"], 3)

    def test_cli_no_args_shows_overview(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code = main([])
            self.assertEqual(code, 0)

    def test_cli_topology_table_and_json(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["topology", "--fabric"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["topology", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 3)

    def test_cli_triage_table_and_json(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["triage", "--pod", "vllm-mistral-7b-tenant-b"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["triage", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

    def test_cli_tune_vllm_table_json_and_infeasible_manifest_guard(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["tune-vllm", "--model", "llama-3.1-70b", "--num-gpus", "4", "--emit-manifest"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["tune-vllm", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertIn("recommended_tp_size", data)
            self.assertTrue(data["fits_in_vram"])

        # Refuses --emit-manifest when requested configuration is infeasible
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            code_bad = main(
                ["tune-vllm", "--model", "llama-3.1-70b", "--dtype", "float32", "--num-gpus", "2", "--emit-manifest"]
            )
            self.assertEqual(code_bad, 1)

    def test_cli_partition_table_and_json(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["partition", "--gpu", "H100-80GB", "--workload", "vllm", "--emit-configmap"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["partition", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertIn("recommended_strategy", data)

    def test_cli_bridge_and_all_12_kubevram_shortcuts_execute(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["bridge", "nvidia-smi", "topo"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_json = main(["bridge", "--format", "json"])
            self.assertEqual(code_json, 0)
            data = json.loads(mock_out.getvalue())
            self.assertEqual(len(data), 12)

        # Every single kubevram_cmd in BRIDGE_KNOWLEDGE_BASE must be an executable CLI command
        for entry in BRIDGE_KNOWLEDGE_BASE:
            tokens = shlex.split(entry["kubevram_cmd"])
            self.assertEqual(tokens[0], "kubevram")
            with patch("sys.stdout", new_callable=io.StringIO):
                rc = main(tokens[1:])
            self.assertEqual(
                rc, 0, f"Bridge command '{entry['kubevram_cmd']}' (id={entry['id']}) failed with exit {rc}"
            )

    def test_cli_rbac_audit_and_fail_on_severity_gates(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            code_table = main(["rbac-audit"])
            self.assertEqual(code_table, 0)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code_gate = main(["rbac-audit", "--format", "json", "--fail-on-severity", "high"])
            self.assertEqual(code_gate, 2)
            data = json.loads(mock_out.getvalue())
            self.assertGreater(len(data), 0)

        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["triage", "--fail-on-severity", "critical"]), 2)
            self.assertEqual(main(["triage", "--pod", "vllm-llama3-70b-fp8-prod-0", "--fail-on-severity", "low"]), 0)
            self.assertEqual(
                main(["partition", "--gpu", "A10G-24GB", "--workload", "embeddings", "--model-vram", "20", "--fail-on-severity", "high"]),
                2,
            )

    def test_cli_live_mode_does_not_silently_fallback_to_mock(self):
        with patch(
            "kubevram.cli.resolve_cluster_state",
            side_effect=CollectorError("kubectl connection refused"),
        ), patch("sys.stderr", new_callable=io.StringIO) as err_out:
            rc = main(["triage", "--live"])
            self.assertEqual(rc, 1)
            self.assertIn("kubectl connection refused", err_out.getvalue())


if __name__ == "__main__":
    unittest.main()
