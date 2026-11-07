#!/usr/bin/env bash
# Fully offline QuantLab demo: measured int8 accuracy/size trade-off plus the
# row-magnitude study, with both gate sides asserted.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== evaluate: int8 matches fp32 accuracy on the fixture, ~3.8x smaller =="
python3 -m quantlab.cli evaluate --max-accuracy-drop 0.0 --min-ratio 3.5
echo

echo "== analyze: per-channel wins where rows differ in magnitude =="
python3 -m quantlab.cli analyze --min-snr-gain 10
echo

echo "== negative control: an impossible size gate fails =="
if python3 -m quantlab.cli evaluate --min-ratio 4.5; then
  echo "ERROR: impossible gate passed" >&2
  exit 1
else
  echo "impossible gate failed as expected"
fi
echo

echo "Demo complete: int8 quantization with measured, gated trade-offs."
