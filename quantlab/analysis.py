"""Where quantization error lives: a controlled row-magnitude study.

Per-tensor quantization uses one scale for the whole matrix, so a single
high-magnitude row forces a coarse step on every other row. Per-channel gives
each row its own scale. This module constructs that exact case - one row
scaled up by a factor - and measures reconstruction error for both modes, so
the advantage of per-channel is demonstrated, not asserted.
"""

from __future__ import annotations

import random

from .quantize import dequantize, error_stats, quantize_per_channel, quantize_tensor


def imbalanced_matrix(rows: int = 8, cols: int = 64, boost: float = 100.0,
                      boosted_row: int = 0, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    m = [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    m[boosted_row] = [w * boost for w in m[boosted_row]]
    return m


def compare_modes(matrix: list[list[float]]) -> dict:
    pt = quantize_tensor(matrix)
    pc = quantize_per_channel(matrix)
    err_pt = error_stats(matrix, dequantize(pt))
    err_pc = error_stats(matrix, dequantize(pc))

    def small_row_snr(packed: dict, exclude: int = 0) -> float:
        """SNR over the un-boosted rows only - the rows that suffer."""
        rest = [row for i, row in enumerate(matrix) if i != exclude]
        rec = [row for i, row in enumerate(dequantize(packed)) if i != exclude]
        return error_stats(rest, rec)["snr_db"]

    return {
        "per_tensor": {**err_pt, "small_row_snr_db": small_row_snr(pt)},
        "per_channel": {**err_pc, "small_row_snr_db": small_row_snr(pc)},
        "small_row_snr_gain_db": round(small_row_snr(pc) - small_row_snr(pt), 2),
    }
