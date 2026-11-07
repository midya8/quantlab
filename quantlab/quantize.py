"""int8 symmetric quantization primitives.

A float tensor is stored as int8 values plus scale factors. Symmetric
quantization maps zero to zero and uses abs(max) as the range:

    scale = absmax / 127
    q = clamp(round(w / scale), -127, 127)
    w_hat = q * scale

Per-tensor uses one scale for the whole matrix; per-channel uses one scale
per row (output channel), which tracks rows whose magnitudes differ widely.
Everything is pure Python: int8 storage is simulated with ints, and size
accounting counts bytes directly.
"""

from __future__ import annotations

import math

Matrix = list[list[float]]


def quantize_tensor(matrix: Matrix) -> dict:
    """Per-tensor symmetric int8 quantization."""
    absmax = max((abs(w) for row in matrix for w in row), default=0.0)
    scale = absmax / 127.0 if absmax > 0 else 1.0
    q = [[max(-127, min(127, round(w / scale))) for w in row] for row in matrix]
    return {"mode": "per-tensor", "q": q, "scales": [scale]}


def quantize_per_channel(matrix: Matrix) -> dict:
    """Per-row symmetric int8 quantization: one scale per output channel."""
    q, scales = [], []
    for row in matrix:
        absmax = max((abs(w) for w in row), default=0.0)
        scale = absmax / 127.0 if absmax > 0 else 1.0
        scales.append(scale)
        q.append([max(-127, min(127, round(w / scale))) for w in row])
    return {"mode": "per-channel", "q": q, "scales": scales}


def dequantize(packed: dict) -> Matrix:
    if packed["mode"] == "per-tensor":
        s = packed["scales"][0]
        return [[v * s for v in row] for row in packed["q"]]
    return [[v * s for v, s in zip(row, [packed["scales"][i]] * len(row))]
            for i, row in enumerate(packed["q"])]


def size_bytes(matrix: Matrix, packed: dict) -> dict:
    params = sum(len(row) for row in matrix)
    return {
        "params": params,
        "fp32_bytes": params * 4,
        "int8_bytes": params + 4 * len(packed["scales"]),
        "ratio": round(params * 4 / (params + 4 * len(packed["scales"])), 3),
    }


def error_stats(original: Matrix, reconstructed: Matrix) -> dict:
    diffs = [abs(a - b) for ra, rb in zip(original, reconstructed) for a, b in zip(ra, rb)]
    flat_o = [w for row in original for w in row]
    flat_r = [w for row in reconstructed for w in row]
    max_err = max(diffs, default=0.0)
    mse = sum(d * d for d in diffs) / max(len(diffs), 1)
    signal = sum(w * w for w in flat_o) / max(len(flat_o), 1)
    snr_db = 10 * math.log10(signal / mse) if mse > 0 and signal > 0 else float("inf")
    dot = sum(a * b for a, b in zip(flat_o, flat_r))
    na = math.sqrt(sum(w * w for w in flat_o))
    nb = math.sqrt(sum(w * w for w in flat_r))
    cos = dot / (na * nb) if na > 0 and nb > 0 else 1.0
    return {"max_abs_error": max_err, "snr_db": round(snr_db, 2), "cosine": round(cos, 6)}
