# QLB1: a small weight-only format

QLB1 is an educational interchange format implemented independently in Python and scalar JavaScript. It is not ONNX, GGUF, a framework checkpoint or a security boundary. It contains no executable code, strings, paths, labels, bias, tokenizer, feature schema or activations.

## Layout

All multibyte fields are little-endian. There is no padding.

| Byte offset | Type | Meaning |
|---|---|---|
| 0 | 4 bytes | ASCII `QLB1` |
| 4 | uint8 | Version, exactly 1 |
| 5 | uint8 | Mode: 0 tensor, 1 output row |
| 6 | uint8 | Bytes per scale: 4 or 8 |
| 7 | uint8 | Reserved flags, exactly 0 |
| 8 | uint32 | Rows, 2 through 8 |
| 12 | uint32 | Columns, 1 through 128 |
| 16 | uint32 | Integer count, exactly rows × columns |
| 20 | uint32 | Scale count, exactly 1 for tensor or rows for row mode |
| 24 | IEEE binary32 or binary64 array | Positive finite scales, at most 1,000,000 |
| following | signed int8 array | Row-major codes in [-127,127]; -128 is rejected |
| final 4 bytes | uint32 | CRC32 of every preceding byte |

Length is exactly `24 + scale_count * scale_bytes + rows * columns + 4`. Maximum size is 1,116 bytes. Parsers reject bad magic, version, flags, counts, shapes, lengths, nonpositive/nonfinite scales, reserved codes, CRC mismatch, truncation and trailing bytes. Counts are checked before decoding arrays. CRC32 is an accidental-corruption check, not authentication: an editor can change valid data and recompute it.

For a matrix W, the unchanged original quantizer computes absmax/127, uses scale 1 for a zero group, rounds division to nearest with ties to even and clips to [-127,127]. QLB1 keeps these integer codes and serializes the scales at the selected precision. Binary32 storage can therefore introduce additional scale error. Codes are not recomputed using the rounded scale. A scale that underflows to zero is rejected. All source input entries must be finite nonboolean binary64 values of magnitude at most 1,000,000.

## Storage is not arithmetic

The original `size_bytes` comparator uses `4*N` for conceptual raw fp32 weights and `N+4*S` for int8 plus binary32 scales. The original Python model actually computes with binary64. This extension leaves that model and its gates unchanged and labels the distinction rather than quietly changing its experiment.

The seed-0, three-output, 64-feature matrix has 192 weights:

| Storage object | Bytes | Ratio against 768 conceptual raw fp32 bytes |
|---|---:|---:|
| Original analytic row-scale payload | 204 | 3.765 (original rounded output) |
| Actual row-scale QLB1, binary32 scales | 232 | 3.310344827586207 |
| Actual row-scale QLB1, binary64 scales | 244 | 3.1475409836065573 |
| Actual tensor-scale QLB1, binary32 scale | 224 | 3.4285714285714284 |
| Actual tensor-scale QLB1, binary64 scale | 228 | 3.3684210526315788 |

These are weights-only comparisons. No RAM, activation, full-checkpoint, latency or integer-kernel claim follows. Dequantized inference uses binary64 products and sequential sums. The export is packed; the Python and browser working arrays are not low-bit RAM measurements.

## Error and decision audit

Let `e = W_hat - W`. For fixed x, a row's logit perturbation obeys `|(W_hat_i-W_i)x| <= sum_k |e_ik| |x_k|` in real arithmetic. A tighter pairwise condition uses `B_aj = sum_k |e_ak-e_jk| |x_k|` for original winner a and competitor j. If original margin `m_aj=(W_a-W_j)x`, the restored margin is at least `m_aj-B_aj`.

All sufficient conditions are computed using exact rational arithmetic over the binary64 inputs and exact stored IEEE scales. For every competitor j, require a positive lower bound when `j<a`, or a nonnegative bound when `a<j`, because the first index wins a tie. This is a sufficient condition only. Failure does not prove that a prediction changes. The exact original and reconstructed predictions are also computed directly.

Python uses `Fraction`; JavaScript uses independent normalized BigInt dyadics. Full rational numerators and denominators are exported; their decimal displays are not the proof. The represented real map uses exact `q*s`, whereas runtime dequantization and summation round operations. Runtime and exact-real predictions are separate outputs. The sufficient condition does not certify runtime accumulation, input uncertainty, unseen data, test accuracy, adversarial robustness or deployment.

The visible flip control `W=[[127,-0.5],[-127,0.5]], x=[0,1]` changes a strict original row-1 winner into a reconstructed tie won by row 0. Reversing the half-unit signs gives a tie that retains row 0 under the explicit tie rule. These are constructed arithmetic controls, not trained experiments. Cancellation and scale-rounding controls are also retained.

## Reproduce

```bash
python -m unittest discover -s tests -v
node --test tests/test_inspector.cjs tests/test_browser.cjs
python scripts/regenerate.py
bash examples/demo.sh
bash examples/verify_wheel.sh
```

The original three quantization/model/analysis modules, three original test files and demo are hash-pinned in `quantlab/data/original-hashes.json`. The README's former test count of 28 was inaccurate: test discovery finds 27 original tests; all 27 remain unchanged. The extension currently has 39 Python tests and four Node/actual-Chrome tests. Fifty-two full report witnesses cover both granularities, both scale precisions, six original test examples, the original eight-row study and all constructed controls. They contain 59,004 numeric values plus every packed byte and exact rational string. Comparison tolerance for binary64 display arithmetic is 1e-12 relative/absolute; byte strings and rational strings must match exactly.

An isolated installed wheel imports all seven implementation modules and runs six CLI commands outside the source checkout, reconstructs the full original train/test data and weights, and checks all 52 full reports. Seven CI jobs cover four Python versions, original demo, browser and installed-package reproduction.

## Attribution and context

These are standard symmetric quantization, ties-to-even rounding, IEEE serialization and elementary linear bounds, not a new quantization algorithm. QLB1 is intentionally small and incompatible with ONNX/GGUF. We do not reproduce their runtimes.

- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://ar5iv.labs.arxiv.org/html/2004.09602): symmetric [-127,127] and row/channel granularity context.
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/): tensor/channel and backend distinctions.
- [Hugging Face quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide): weight versus activation quantization and metadata granularity.
- [Intel Neural Network Distiller](https://intellabs.github.io/distiller/algo_quantization.html): symmetric scale and output-channel context.
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html): nearest-even rounding, precision and granularity context, not format compatibility.
- [Python struct](https://docs.python.org/3/library/struct.html) and [round](https://docs.python.org/3/library/functions.html): standard-endian IEEE packing and Python tie semantics.
