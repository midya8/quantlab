# QuantLab

Post-training weight quantization from scratch, with actual packed artifacts and an independent browser inspector. Follow a weight through its int8 code, stored IEEE scale, reconstructed logit and exact fixed-input decision-margin audit.

Open `docs/index.html` locally without a server. Python and browser arithmetic are independent. All original training, tests, magnitude controls and gates remain unchanged.

## Real bytes, explicit arithmetic

The original seed-0 perceptron and its quantized versions reach 59/60 synthetic test accuracy. That does not imply parity on unseen data. The original row-magnitude study remains: tensor-scale small-row SNR 9.25 dB versus row-scale 48.39 dB, a 39.14 dB difference.

The original analytic row-scale payload is 204 B, compared with 768 conceptual fp32 weight bytes. That is not the size of a real file. QLB1 now stores row-major signed bytes, IEEE scales, a 24-byte versioned header and CRC32:

| Original 192 weights | Actual file bytes | vs conceptual 768 B raw fp32 |
|---|---:|---:|
| Row scales, binary32 | 232 | 3.3103x |
| Row scales, binary64 | 244 | 3.1475x |
| Tensor scale, binary32 | 224 | 3.4286x |
| Tensor scale, binary64 | 228 | 3.3684x |

The original weight-only source and inference use binary64, not fp32. That inspector measures no integer matmul, RAM or speedup. The separate activation extension below adds bounded integer dot products, not a hardware benchmark. Packing weights does not make the floating-point working arrays low-bit. Scale precision changes only stored metadata; original int8 codes remain fixed. CRC32 is not authentication.

## Inspect every consequence

- All 60 original test examples, both quantization granularities, both scale precisions and the original eight-row magnitude study remain visible. The archive also contains all 120 training examples. There is no best-example selection or retraining from the controls.
- Editable bounded matrices and inputs expose every source weight, code, scale, reconstructed weight and signed contribution to the logit error. Download the input, full report or real QLB1 file; import a file through the independent strict browser parser.
- Python `Fraction` and browser BigInt dyadics compute exact represented-real logits and pairwise sufficient margin bounds. Full rational witnesses are exported. Runtime predictions are separate: the exact audit does not certify floating-point accumulation, uncertain inputs, unseen examples, accuracy or deployment.
- A visible half-step counterexample changes a strict winner into a tie lost by the original winner. The inverse tie case stays correct under the explicit first-index rule. A failed sufficient condition does not imply a changed prediction.

## Run

```bash
python -m quantlab.cli evaluate --max-accuracy-drop 0 --min-ratio 3.5
python -m quantlab.cli analyze --min-snr-gain 10
python -m quantlab.cli json
# JSON schema: {"weights": [[...], [...]], "x": [...]}
python -m quantlab.cli inspect --input input.json --mode per-channel --scale-bits 64
python -m quantlab.cli pack --input input.json --output weights.qlb --scale-bits 32
python -m quantlab.cli unpack --input weights.qlb
python -m unittest discover -s tests -v
node --test tests/test_inspector.cjs tests/test_browser.cjs
bash examples/demo.sh
bash examples/verify_wheel.sh
```

`pack` refuses to overwrite an existing file. Inputs are bounded to 2..8 output rows, 1..128 features and finite nonboolean numbers with magnitude at most one million. The JSON reader rejects duplicate keys, nonfinite values, deep nesting, unknown fields and trailing content. QLB1 rejects malformed headers, inconsistent counts, reserved -128 codes, invalid scales, oversized files, truncation, trailing bytes and CRC mismatch. See [FORMAT.md](FORMAT.md) for the byte-level contract and error-bound derivation.

## Verification

The earlier packed-weight extension had 39 Python tests, including all 27 unchanged original tests. The old README's claim of 28 original tests was a counting error, not a removed test. Four Node/actual-Chrome tests compare 52 complete native/browser reports containing 59,004 numbers, every exported byte and exact rational witnesses. Original data, trainer and gates are preserved. Tests cover every single-byte corruption and every truncation of a small file, malformed files even with recomputed CRC, exact tie rules, randomized independent rational oracles, all original examples, strict invalid-state non-revival, imports, downloads, keyboard use and responsive layouts.

The earlier isolated installed wheel imported seven modules and ran six commands and reconstructs the complete original fixture outside the checkout. The current eight CI jobs cover Python 3.10-3.13, original demo, actual Chrome, installed reproduction and an actual NumPy int32 reference.

## Limits and attribution

This is a finite synthetic classifier, not a transformer or deployment benchmark. Hashed bag-of-words features discard word order and can collide. The file contains weights only, no bias, feature extraction, labels or activations. The fixed-input bound says nothing about uncertainty or unobserved data. Standard symmetric quantization and linear inequalities are attributed, not presented as new methods. QLB1 is a small educational format, not ONNX, GGUF or a secure checkpoint. See [methods, assumptions and sources](FORMAT.md).

MIT.


## Activation × weight integer dot-product audit

A separate [interactive page](https://midya8.github.io/quantlab/integer.html) traces signed 8-bit products and every partial accumulator, with fixed user-declared activation range, stored binary32/64 activation scales, exact nearest-even codes and clipping flags. Python `Fraction` and independent browser BigInt dyadics export the complete exact decomposition `weight error + activation error = total error`. Every prefix fits int32 by the checked shape bound, up to 2,064,512. Integer products are exact; final rescale is floating, not integer-only end-to-end inference.

The visible activation-only control uses exactly represented weights `[[127,0],[0,127]]`, input `[0.4,0.49]`, and range `127`: the strict row-1 winner becomes a zero-code tie won by row 0, with exactly zero weight error. This is constructed, not a trained result or accuracy claim.

```sh
quantlab integer --input request.json --mode per-channel --scale-bits 64 --activation-bits 32
python scripts/integer_fixtures.py --check
PYTHONPATH=. python scripts/verify_integer.py
```

[INTEGER.md](INTEGER.md) specifies semantics, limits, provenance and reproduction. Original QLB1 bytes, training, 39 earlier Python tests and 52 earlier report witnesses stay unchanged. Current totals: 47 Python tests and seven Node/actual-Chrome tests. New checks include 240 separately rounded rational reference cases, 519 signed half-step activation cases, 72 full cross-runtime reports covering 6,848 products, NumPy int32 dot products/prefixes, 168 browser row selections and 20 responsive views. An isolated installed package verifies every new report and both CLI entry points. No hardware speed, RAM, calibrated range, unseen accuracy or new-algorithm claim.
