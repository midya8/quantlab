# Activation × weight integer dot-product audit

## Semantics

The integer inspector quantizes activations as well as weights, then exposes
every integer product and partial sum. It uses the original QLB1 weights without
changing the format, trainer or weight-only inspector. Final score rescaling
uses floating point; this is not integer-only end-to-end inference.

 The new request has exactly `weights`, `x`, and `activation_range`. Shape: 2..8 rows and 1..128 features. All values are nonboolean finite binary64 numbers of absolute magnitude at most 1,000,000; range must be positive. JSON rejects duplicate keys, nonfinite constants, excessive nesting and inputs above 200KB. Direct native calls validate the same numeric and shape limits. Unrepresentable positive scales are rejected, never silently replaced.

The fixed declared activation range r produces s_x = r / 127, first divided in binary64, then rounded to the selected binary32/64 storage precision. Let s be that exact stored value. Each code is clamp(round_even(exact(x) / exact(s)), -127,127). This activation coding uses exact represented-real division, not a rounded binary64 quotient. Original weight codes retain the old inspector's binary64 division/rounding and QLB1 serialization semantics. The differing rounding paths are deliberate and explicit, not ONNX or TFLite compatibility.

Zero-point is zero. Every restored activation is exactly q_x s_x. The endpoint flag is |x| > 127 s_x over exact rationals, independent of rounded-code saturation. Metadata rounding can move the endpoint slightly from the requested range; both are exported. The range is user-specified, not calibrated from test data.

Each integer row computes products q_w q_x and left-to-right integer prefixes. Since |q_w|, |q_x| <=127, each product <=16,129 in magnitude and each prefix <= n*16,129 <=2,064,512 <2^31-1. Python uses unbounded integers and JavaScript uses exact bounded Numbers. This is not a machine-int32 performance claim. Actual NumPy int32 matmul and cumsum are an independent arithmetic check.

## Exact error decomposition

Let a_i = W_i x, b_i = W_hat_i x, c_i = W_hat_i x_hat = s_wi s_x sum(q_wi q_x). For the exact represented-real maps:

- Weight contribution per coordinate: (W_hat_ik - W_ik) x_k.
- Activation contribution: W_hat_ik (x_hat_k - x_k), including clipping.
- Summing yields b_i-a_i and c_i-b_i, whose sum is c_i-a_i exactly.

This is an order-dependent algebraic decomposition, not causal attribution. Reversing the order changes the terms. Full rational strings and every coordinate contribution are retained. Python Fraction and independent JS BigInt dyadics agree exactly on all strings. Decimal displays are approximate and are not proofs.

Exact-real predictions and binary64-rescaled runtime predictions are separately exported. First index wins ties. Runtime rescale order is `(integer_accumulator * stored_weight_scale) * stored_activation_scale`. No guarantee is made that floating rescale equals a different dequantized floating summation order.

## Controls and checks

Nine controls cover an activation-only decision change, fixed-range clipping, exact signed half-way ties, zeros, maximum accumulator shape, scale precision, and three original trained test rows. Every combination of two weight granularities, two weight scale precisions and two activation scale precisions gives 72 full reports and 6,848 products. The original trained fixture and previous 52 witnesses are retained byte-for-byte.

The activation-only control has W=[[127,0],[0,127]], x=[0.4,0.49], r=127. Stored weight and activation scales are exactly 1. Source scores are the represented-real values 127*x, with row 1 strictly largest. Quantized x=[0,0], so both accumulators are zero and row 0 wins the tie. Weight error is exactly zero. This example is constructed, not learned or an accuracy measurement.

Native tests separately compute floor/remainder nearest-even activation codes for 240 seeded random reports, check every product/prefix and exact decomposition, and exhaust all 519 half-unit values from -129.5 to 129.5. Other tests cover both bound signs, underflow rejection, duplicate keys, types, no input mutation, old-payload equality and CLI failure behavior. The current suite has 47 Python tests, including all 39 earlier tests, and seven Node/actual-Chrome tests, including all four earlier tests. Browser tests verify all 72 reports, 168 row selections and 20 views at 1280/768/390/320 widths, trusted report/input downloads, duplicate-key rejection and edit invalidation.

## Reproduce

```sh
python -m unittest discover -s tests -v
python scripts/integer_fixtures.py --check
PYTHONPATH=. python scripts/verify_integer.py
node --test tests/test_inspector.cjs tests/test_browser.cjs tests/test_integer.cjs tests/test_integer_browser.cjs
bash examples/verify_wheel.sh
```

The wheel is installed into a fresh isolated environment. Outside the source checkout it reproduces original training and 52 reports, imports the eighth module, verifies all 72 new reports, and executes the integer command through both installed entry points. The optional NumPy check runs when NumPy is available; it is not a runtime dependency and the isolated package deliberately uses only the standard library. Package and CI test totals are not hardware benchmarks.

## Prior work and limits

- [Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877), 2017/2018, primary paper: low-bit weight/activation operands, int32 accumulation and fixed-point output requantization. This project implements only symmetric dot products and floating final rescale, not their full integer-only inference framework or training procedure.
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html), operator specification: scale/zero-point and nearest-even context, not compatibility.
- [PyTorch quantization in practice](https://pytorch.org/blog/quantization-in-practice/), official technical guidance: quantization ranges, tensor/channel distinctions and backend-specific behavior.
- [Hugging Face quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide), official technical guide: weights versus activations and storage granularity.
- [Python fractions](https://docs.python.org/3/library/fractions.html), standard library reference: exact representation of binary floats and Fraction nearest-even rounding.
- [Python struct](https://docs.python.org/3/library/struct.html), standard library reference: explicit IEEE scale storage.
- [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), official reference: separate array integer multiplication check.

No bias, asymmetric zero-point, integer output requantization, nonlinear network, hardware kernel, latency, memory-usage benchmark, deployment proof, calibrated range, learned improvement, adversarial bound, accuracy guarantee or new algorithm is supplied. Public weights, constructed controls and synthetic original fixture only.
