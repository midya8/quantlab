import unittest

from quantlab.quantize import (
    dequantize, error_stats, quantize_per_channel, quantize_tensor, size_bytes,
)


class QuantizeTests(unittest.TestCase):
    def setUp(self):
        self.m = [[0.5, -1.0, 2.0], [0.01, -0.02, 0.03]]

    def test_int8_range(self):
        for mode in (quantize_tensor, quantize_per_channel):
            q = mode(self.m)["q"]
            self.assertTrue(all(-127 <= v <= 127 for row in q for v in row))

    def test_absmax_maps_to_full_range(self):
        packed = quantize_tensor(self.m)  # absmax = 2.0, symmetric: only absmax hits +-127
        self.assertEqual(max(v for row in packed["q"] for v in row), 127)
        self.assertEqual(min(v for row in packed["q"] for v in row), -64)  # -1.0 / (2/127)

    def test_roundtrip_close(self):
        packed = quantize_per_channel(self.m)
        rec = dequantize(packed)
        err = error_stats(self.m, rec)
        self.assertLess(err["max_abs_error"], 0.02)
        self.assertGreater(err["cosine"], 0.9999)

    def test_zero_maps_to_zero(self):
        packed = quantize_tensor([[0.0, 1.0]])
        self.assertEqual(packed["q"][0][0], 0)
        self.assertEqual(dequantize(packed)[0][0], 0.0)

    def test_all_zero_matrix_safe(self):
        packed = quantize_tensor([[0.0, 0.0]])
        self.assertEqual(dequantize(packed), [[0.0, 0.0]])

    def test_per_channel_has_scale_per_row(self):
        packed = quantize_per_channel(self.m)
        self.assertEqual(len(packed["scales"]), 2)
        self.assertEqual(len(quantize_tensor(self.m)["scales"]), 1)


class SizeTests(unittest.TestCase):
    def test_four_x_at_scale(self):
        m = [[0.1] * 100 for _ in range(100)]
        size = size_bytes(m, quantize_per_channel(m))
        self.assertEqual(size["fp32_bytes"], 40000)
        self.assertGreater(size["ratio"], 3.8)

    def test_counts_scales(self):
        m = [[0.1] * 10]
        self.assertEqual(size_bytes(m, quantize_tensor(m))["int8_bytes"], 14)
        self.assertEqual(size_bytes(m, quantize_per_channel(m))["int8_bytes"], 14)


class ErrorStatsTests(unittest.TestCase):
    def test_identical_is_lossless(self):
        m = [[1.0, 2.0]]
        e = error_stats(m, [row[:] for row in m])
        self.assertEqual(e["max_abs_error"], 0.0)
        self.assertEqual(e["cosine"], 1.0)

    def test_detects_error(self):
        e = error_stats([[1.0]], [[1.5]])
        self.assertAlmostEqual(e["max_abs_error"], 0.5)


if __name__ == "__main__":
    unittest.main()
