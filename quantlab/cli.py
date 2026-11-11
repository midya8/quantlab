"""Command-line interface for QuantLab."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import compare_modes, imbalanced_matrix
from .model import LinearClassifier, build_dataset
from .quantize import dequantize, error_stats, quantize_per_channel, quantize_tensor, size_bytes


def _train_fixture(seed: int, dim: int, epochs: int):
    train, topics = build_dataset(seed=seed, per_topic=40)
    test, _ = build_dataset(seed=seed + 1, per_topic=20)
    clf = LinearClassifier(dim=dim, classes=3).train(train, epochs=epochs)
    return clf, train, test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ev = sub.add_parser("evaluate", help="fp32 vs int8 accuracy and size on the fixture")
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--dim", type=int, default=64)
    p_ev.add_argument("--epochs", type=int, default=5)
    p_ev.add_argument("--mode", choices=["per-tensor", "per-channel"], default="per-channel")
    p_ev.add_argument("--max-accuracy-drop", type=float, default=None,
                      help="exit 1 if int8 test accuracy drops more than this vs fp32")
    p_ev.add_argument("--min-ratio", type=float, default=None,
                      help="exit 1 if the fp32/int8 size ratio is below this")

    p_an = sub.add_parser("analyze", help="per-tensor vs per-channel on a row-magnitude study")
    p_an.add_argument("--boost", type=float, default=100.0)
    p_an.add_argument("--min-snr-gain", type=float, default=None,
                      help="exit 1 if per-channel small-row SNR gain (dB) is below this")

    p_json = sub.add_parser("json", help="full report as JSON")
    p_json.add_argument("--seed", type=int, default=0)
    p_json.add_argument("--dim", type=int, default=64)
    p_json.add_argument("--epochs", type=int, default=5)

    p_in = sub.add_parser("inspect", help="full artifact and exact-dyadic margin audit")
    p_in.add_argument("--input", required=True, help="bounded weights/x JSON")
    p_in.add_argument("--mode", choices=["per-tensor", "per-channel"], default="per-channel")
    p_in.add_argument("--scale-bits", choices=[32,64], type=int, default=64)
    p_pack = sub.add_parser("pack", help="write a new QLB1 weight file, never overwrite")
    p_pack.add_argument("--input", required=True)
    p_pack.add_argument("--output", required=True)
    p_pack.add_argument("--mode", choices=["per-tensor", "per-channel"], default="per-channel")
    p_pack.add_argument("--scale-bits", choices=[32,64], type=int, default=64)
    p_unpack = sub.add_parser("unpack", help="validate QLB1 and print metadata and weights")
    p_unpack.add_argument("--input", required=True)
    p_int = sub.add_parser("integer", help="activation codes, integer products and exact error decomposition")
    p_int.add_argument("--input", required=True)
    p_int.add_argument("--mode", choices=["per-tensor","per-channel"],default="per-channel")
    p_int.add_argument("--scale-bits",type=int,choices=[32,64],default=64)
    p_int.add_argument("--activation-bits",type=int,choices=[32,64],default=64)
    args = parser.parse_args(argv)
    if args.cmd == "integer":
        from .integer import integer_audit,read_request
        try:
            with open(args.input,encoding="utf-8")as f:v=read_request(f.read(200001))
            print(json.dumps(integer_audit(v,args.mode,args.scale_bits,args.activation_bits),indent=2,allow_nan=False));return 0
        except (OSError,ValueError,OverflowError)as e:
            print("ERROR: "+str(e),file=sys.stderr);return 2
    if args.cmd in ("inspect", "pack", "unpack"):
        from .inspector import strict_json, inspect
        from .packed import decode, MAX_BYTES
        try:
            if args.cmd == "unpack":
                with open(args.input, "rb") as f:
                    result = decode(f.read(MAX_BYTES + 1))
            else:
                with open(args.input, "r", encoding="utf-8") as f:
                    text = f.read(200001)
                result = inspect(strict_json(text), args.mode, args.scale_bits)
                if args.cmd == "pack":
                    with open(args.output, "xb") as f:
                        f.write(bytes.fromhex(result["payload_hex"]))
                    result = {"output":args.output,"layout":result["layout"],"scale_bits":args.scale_bits}
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        except (OSError, ValueError, OverflowError) as e:
            print("ERROR: " + str(e), file=sys.stderr)
            return 2


    if args.cmd == "evaluate":
        clf, train, test = _train_fixture(args.seed, args.dim, args.epochs)
        q = clf.quantized(args.mode)
        fp32_acc = clf.accuracy(test)
        int8_acc = q.accuracy(test)
        size = size_bytes(clf.weights, q.packed)
        err = error_stats(clf.weights, dequantize(q.packed))
        print(f"fixture: seed={args.seed} dim={args.dim} mode={args.mode}")
        print(f"test accuracy: fp32 {fp32_acc:.4f} | int8 {int8_acc:.4f} | drop {fp32_acc - int8_acc:+.4f}")
        print(f"weights: {size['params']} params | fp32 {size['fp32_bytes']} B | int8 {size['int8_bytes']} B ({size['ratio']}x smaller)")
        print(f"reconstruction: max abs error {err['max_abs_error']:.5f} | SNR {err['snr_db']} dB | cosine {err['cosine']}")
        failed = []
        if args.max_accuracy_drop is not None and fp32_acc - int8_acc > args.max_accuracy_drop:
            failed.append(f"accuracy drop {fp32_acc - int8_acc:+.4f} > {args.max_accuracy_drop:.4f}")
        if args.min_ratio is not None and size["ratio"] < args.min_ratio:
            failed.append(f"size ratio {size['ratio']}x < {args.min_ratio}x")
        if failed:
            print("GATE FAILED: " + "; ".join(failed))
            return 1
        return 0

    if args.cmd == "analyze":
        matrix = imbalanced_matrix(boost=args.boost)
        rep = compare_modes(matrix)
        pt, pc = rep["per_tensor"], rep["per_channel"]
        print(f"one row boosted x{args.boost:g} among small rows:")
        print(f"per-tensor:  small-row SNR {pt['small_row_snr_db']} dB | overall SNR {pt['snr_db']} dB")
        print(f"per-channel: small-row SNR {pc['small_row_snr_db']} dB | overall SNR {pc['snr_db']} dB")
        print(f"per-channel small-row gain: {rep['small_row_snr_gain_db']} dB")
        if args.min_snr_gain is not None and rep["small_row_snr_gain_db"] < args.min_snr_gain:
            print(f"GATE FAILED: gain {rep['small_row_snr_gain_db']} dB < {args.min_snr_gain} dB")
            return 1
        return 0

    if args.cmd == "json":
        clf, train, test = _train_fixture(args.seed, args.dim, args.epochs)
        out = {"config": {"seed": args.seed, "dim": args.dim, "epochs": args.epochs}}
        for mode in ("per-tensor", "per-channel"):
            q = clf.quantized(mode)
            out[mode] = {
                "test_accuracy": q.accuracy(test),
                "fp32_test_accuracy": clf.accuracy(test),
                "size": size_bytes(clf.weights, q.packed),
                "error": error_stats(clf.weights, dequantize(q.packed)),
            }
        out["magnitude_study"] = compare_modes(imbalanced_matrix())
        print(json.dumps(out, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
