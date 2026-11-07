import io
import json
import unittest
from contextlib import redirect_stdout

from quantlab.cli import main


def run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


class EvaluateTests(unittest.TestCase):
    def test_reports_accuracy_and_size(self):
        code, out = run_cli(["evaluate"])
        self.assertEqual(code, 0)
        self.assertIn("test accuracy:", out)
        self.assertIn("x smaller", out)

    def test_gates_pass_at_pinned_levels(self):
        code, out = run_cli(["evaluate", "--max-accuracy-drop", "0.0", "--min-ratio", "3.5"])
        self.assertEqual(code, 0, out)

    def test_impossible_gate_fails(self):
        code, out = run_cli(["evaluate", "--min-ratio", "4.5"])
        self.assertEqual(code, 1)
        self.assertIn("GATE FAILED", out)


class AnalyzeTests(unittest.TestCase):
    def test_gate_passes(self):
        code, out = run_cli(["analyze", "--min-snr-gain", "10"])
        self.assertEqual(code, 0, out)
        self.assertIn("small-row gain", out)

    def test_gate_fails(self):
        code, _ = run_cli(["analyze", "--min-snr-gain", "60"])
        self.assertEqual(code, 1)


class JsonTests(unittest.TestCase):
    def test_report_shape(self):
        code, out = run_cli(["json"])
        self.assertEqual(code, 0)
        r = json.loads(out)
        for key in ("per-tensor", "per-channel", "magnitude_study", "config"):
            self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
