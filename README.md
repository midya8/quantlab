# QuantLab

Post-training int8 quantization built from scratch in pure Python: symmetric quantization with per-tensor and per-channel scales, byte-exact size accounting, reconstruction error metrics, and a trained fixture model whose accuracy cost is measured rather than assumed.

QuantLab answers the questions quantization write-ups usually skip:

- **What does int8 storage cost in accuracy?** A tiny linear classifier is trained in float on a deterministic synthetic dataset (one seed reproduces the run bit for bit), then its weights are quantized to int8. On the checked-in fixture, per-channel int8 matches fp32 test accuracy exactly (drop 0.0000) at 3.765x smaller storage.
- **Where does quantization error actually live?** A controlled study boosts one matrix row x100: per-tensor's single scale forces a coarse step on every small row (small-row SNR 9.25 dB), while per-channel's per-row scales hold 48.39 dB - a 39.14 dB measured gain. This is the case per-channel quantization exists for, demonstrated directly.

## Measured results

Fixture (seed 0, dim 64, 3 classes, per-channel), from `quantlab evaluate`:

| metric | fp32 | int8 |
|---|---|---|
| test accuracy | 0.9833 | 0.9833 (drop 0.0000) |
| weight storage | 768 B | 204 B (3.765x) |
| reconstruction | - | SNR 45.31 dB, cosine 0.999988 |

Row-magnitude study (`quantlab analyze`): per-channel small-row SNR gain +39.14 dB over per-tensor.

CI pins both: `--max-accuracy-drop 0.0 --min-ratio 3.5` and `--min-snr-gain 10`.

## How it works

Symmetric quantization maps zero to zero and uses abs(max) as the range: `scale = absmax / 127`, `q = clamp(round(w / scale), -127, 127)`, `w_hat = q * scale`. Per-tensor uses one scale per matrix; per-channel one scale per row. int8 storage is simulated with Python ints (values provably inside [-127, 127] in the test suite) and size accounting counts bytes directly: params x 4 vs params + 4 x scales. The quantized classifier stores int8 weights plus scales and dequantizes for inference - the shape a real quantized deployment takes.

## Usage

```bash
python3 -m quantlab.cli evaluate --max-accuracy-drop 0.0 --min-ratio 3.5
python3 -m quantlab.cli analyze --min-snr-gain 10
python3 -m quantlab.cli json        # both modes + the magnitude study as JSON
bash examples/demo.sh               # offline demo, both gate sides
```

## Layout

- `quantlab/quantize.py` - quantization primitives, dequantize, size accounting, error metrics (max abs error, SNR dB, cosine).
- `quantlab/model.py` - deterministic dataset, hashed bag-of-words features, perceptron-trained linear classifier, int8 quantized variant.
- `quantlab/analysis.py` - the row-magnitude study comparing modes.
- `tests/` - 28 unit tests covering quantization ranges and round-trips, zero handling, size accounting, dataset determinism, fixture accuracy parity, both analysis sides, and every CLI gate.

## Limits

- int8 storage is simulated, not packed bits: values are Python ints proven in range, and size accounting is byte arithmetic, not a serialized format.
- Post-training quantization only; no quantization-aware training, no int4/mixed precision, no activation quantization.
- The fixture model is tiny by design so the accuracy measurement is exact and fast; real models show non-zero drops, and nothing here claims they would not.
- The classifier uses Python's perceptron over hashed features - enough to measure weight-quantization cost honestly, not a claim about LLM-scale behavior.
