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

The Python source and inference use binary64, not fp32. No integer matmul, RAM or speedup is measured. Packing weights does not make the floating-point working arrays low-bit. Scale precision changes only stored metadata; original int8 codes remain fixed. CRC32 is not authentication.

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

39 Python tests include all 27 unchanged original tests. The old README's claim of 28 original tests was a counting error, not a removed test. Four Node/actual-Chrome tests compare 52 complete native/browser reports containing 59,004 numbers, every exported byte and exact rational witnesses. Original data, trainer and gates are preserved. Tests cover every single-byte corruption and every truncation of a small file, malformed files even with recomputed CRC, exact tie rules, randomized independent rational oracles, all original examples, strict invalid-state non-revival, imports, downloads, keyboard use and responsive layouts.

An isolated installed wheel imports all seven modules, runs all six commands and reconstructs the complete original fixture outside the checkout. Seven CI jobs cover Python 3.10-3.13, original demo, actual Chrome and installed reproduction.

## Limits and attribution

This is a finite synthetic classifier, not a transformer or deployment benchmark. Hashed bag-of-words features discard word order and can collide. The file contains weights only, no bias, feature extraction, labels or activations. The fixed-input bound says nothing about uncertainty or unobserved data. Standard symmetric quantization and linear inequalities are attributed, not presented as new methods. QLB1 is a small educational format, not ONNX, GGUF or a secure checkpoint. See [methods, assumptions and sources](FORMAT.md).

MIT.
