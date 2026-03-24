#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep YTILE x UNRL tuning parameters for the fp16/bf16 skinny GEMM kernel.

Sweeps all (YTILE, UNRL) combinations for each shape and batch size,
producing a CSV of results and summary tables comparing against the
current heuristic.

Usage:
    python sweep_fp16_kernel.py
    python sweep_fp16_kernel.py --batch-sizes 1 4
    python sweep_fp16_kernel.py --dtype bfloat16
    python sweep_fp16_kernel.py --shapes 4096x4096
"""

import argparse
import csv
import itertools
import time

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units as get_cu_count

SHAPES = [
    (9728, 896, "Qwen0.5B gate_up"),
    (2048, 2048, "Gemma2B q/o"),
    (2560, 2048, "Gemma2B qkv"),
    (32768, 2048, "Gemma2B gate_up"),
    (2048, 16384, "Gemma2B down"),
    (4096, 2048, "Qwen1.7B qkv"),
    (12288, 2048, "Qwen1.7B gate_up"),
    (2048, 6144, "Qwen1.7B down"),
    (2560, 2560, "Qwen3-4B q/o"),
    (6144, 2560, "Qwen3-4B qkv"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    (151936, 2560, "Qwen3-4B lm_head"),
    (22016, 2048, "Qwen2.5VL-3B gate_up"),
    (2048, 11008, "Qwen2.5VL-3B down"),
    (3584, 3584, "Qwen7B q/o"),
    (4608, 3584, "Qwen7B qkv"),
    (37888, 3584, "Qwen7B gate_up"),
    (3584, 18944, "Qwen7B down"),
    (152064, 3584, "Qwen7B lm_head"),
    (4096, 4096, "LLaMA8B q/o"),
    (6144, 4096, "LLaMA8B qkv"),
    (28672, 4096, "LLaMA8B gate_up"),
    (4096, 14336, "LLaMA8B down"),
    (11008, 4096, "LLaMA2-7B up/gate"),
    (22016, 4096, "LLaMA2-7B gate_up"),
    (4096, 11008, "LLaMA2-7B down"),
]

YTILES = [1, 2, 3, 4]
UNRLS = [1, 2, 4]

LDS_CAPACITY = 64 * 1024 // 2


def heuristic_config(M, K, N, cu_count):
    """Replicate the C++ WVSPLIT_TILE gfx11 heuristic.

    Tuned for RDNA 3.5 with CuCount = num_CUs (40 on Strix Halo).
    """
    sYT = (M + cu_count * 4 - 1) // (cu_count * 4)

    if sYT <= 1:
        return 1, 4
    elif sYT <= 13:
        return (1, 4) if K <= 2048 else (1, 1)
    elif K <= 2048:
        if sYT <= 16:
            return 1, 2
        elif sYT <= 26:
            return 1, 4
        else:
            return 1, 1
    elif K <= 3584:
        if sYT >= 237:
            return 4, 1
        elif sYT <= 39 and M % 3 == 0:
            return 3, 2
        else:
            return 1, 2
    else:
        # K >= 4096
        if sYT >= 237:
            return 4, 1
        else:
            return 1, 1


def parse_shape(s):
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def run_sweep(shapes, batch_sizes, dtype, warmup, rep):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    GIB = 1024**3

    total_combos = len(shapes) * len(batch_sizes) * len(YTILES) * len(UNRLS)
    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}, dtype: {dtype}")
    print(f"Param grid: YTILE={YTILES} x UNRL={UNRLS}")
    print(f"Max combos (before filtering): {total_combos}")
    print(f"warmup={warmup}, rep={rep}")
    print()

    dtype_tag = "fp16" if dtype == torch.float16 else "bf16"
    csv_path = f"/scratch/mgehre/tmp/{dtype_tag}_sweep_results.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "M",
                "K",
                "N",
                "label",
                "ytile",
                "unrl",
                "time_us",
                "total_bw_gibs",
            ]
        )

        results = []
        skipped = 0
        tested = 0
        t0 = time.time()

        for M, K, label in shapes:
            for N in batch_sizes:
                weight = (torch.rand(M, K, dtype=dtype, device="cuda") - 0.5) * 0.01
                activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01
                total_bytes = (M * K + N * K + N * M) * 2

                shape_results = []
                for ytile, unrl in itertools.product(YTILES, UNRLS):
                    if M % ytile != 0:
                        skipped += 1
                        continue

                    try:

                        def _bench(w=weight, a=activation, yt=ytile, ur=unrl):
                            return ops.wvSplitK_sweep(
                                w,
                                a.view(-1, a.size(-1)),
                                cu_count,
                                yt,
                                ur,
                            )

                        ms = triton.testing.do_bench(
                            _bench,
                            warmup=warmup,
                            rep=rep,
                            return_mode="median",
                        )
                        time_us = ms * 1000
                        bw_gibs = total_bytes / (ms * 1e-3) / GIB
                    except Exception as e:
                        time_us = float("inf")
                        bw_gibs = 0.0
                        print(f"  ERROR: N={N} {M}x{K} yt={ytile} ur={unrl}: {e}")

                    writer.writerow(
                        [
                            M,
                            K,
                            N,
                            label,
                            ytile,
                            unrl,
                            f"{time_us:.1f}",
                            f"{bw_gibs:.1f}",
                        ]
                    )
                    shape_results.append(
                        {
                            "M": M,
                            "K": K,
                            "N": N,
                            "label": label,
                            "ytile": ytile,
                            "unrl": unrl,
                            "time_us": time_us,
                            "bw_gibs": bw_gibs,
                        }
                    )
                    tested += 1

                if shape_results:
                    best = min(shape_results, key=lambda r: r["time_us"])
                    results.append(best)
                    elapsed = time.time() - t0
                    print(
                        f"  N={N} {M:>6}x{K:<6} {label:<22} "
                        f"best: yt={best['ytile']}"
                        f" ur={best['unrl']}  "
                        f"{best['time_us']:>8.1f} us"
                        f"  {best['bw_gibs']:>6.1f} GiB/s  "
                        f"[{tested} tested, {elapsed:.0f}s elapsed]"
                    )

                csv_file.flush()
    elapsed = time.time() - t0
    print()
    print(f"Done: {tested} combos tested, {skipped} skipped, {elapsed:.0f}s total")
    print(f"Full CSV: {csv_path}")
    print()

    analyze_results(csv_path, results, cu_count)


def analyze_results(csv_path, best_per_shape, cu_count):
    """Analyze sweep results and print summary tables."""
    import collections

    all_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["time_us"] = float(row["time_us"])
            row["total_bw_gibs"] = float(row["total_bw_gibs"])
            row["M"] = int(row["M"])
            row["K"] = int(row["K"])
            row["N"] = int(row["N"])
            row["ytile"] = int(row["ytile"])
            row["unrl"] = int(row["unrl"])
            all_rows.append(row)

    print("=" * 110)
    print("BEST CONFIG PER SHAPE")
    print("=" * 110)
    print(
        f"{'N':>2} {'M':>6}x{'K':<6} {'Label':<22} "
        f"{'yt':>3} {'ur':>3} "
        f"{'time_us':>9} {'BW GiB/s':>9}"
    )
    print("-" * 110)
    for r in best_per_shape:
        print(
            f"{r['N']:>2} {r['M']:>6}x{r['K']:<6} {r['label']:<22} "
            f"{r['ytile']:>3} {r['unrl']:>3} "
            f"{r['time_us']:>9.1f} {r['bw_gibs']:>9.1f}"
        )

    print()
    print("=" * 110)
    print("YTILE/UNRL ANALYSIS (best per shape vs current heuristic)")
    print("=" * 110)

    shape_keys = set()
    for row in all_rows:
        shape_keys.add((row["M"], row["K"], row["N"], row["label"]))

    total_heur_time = 0.0
    total_best_time = 0.0
    regressions = []

    for sk in sorted(shape_keys):
        M, K, N, label = sk
        rows_for_shape = [
            r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
        ]
        if not rows_for_shape:
            continue
        best = min(rows_for_shape, key=lambda r: r["time_us"])

        heur_yt, heur_ur = heuristic_config(M, K, N, cu_count)

        heur_rows = [
            r for r in rows_for_shape if r["ytile"] == heur_yt and r["unrl"] == heur_ur
        ]
        heur_best = min(heur_rows, key=lambda r: r["time_us"]) if heur_rows else None

        total_best_time += best["time_us"]

        marker = ""
        if heur_best:
            total_heur_time += heur_best["time_us"]
            if heur_best["time_us"] > best["time_us"] * 1.05:
                regret = (
                    (heur_best["time_us"] - best["time_us"]) / best["time_us"] * 100
                )
                marker = f"  <-- heuristic {regret:.0f}% slower"
                regressions.append((sk, regret, best, heur_best))
        else:
            total_heur_time += best["time_us"]

        heur_str = f" ({heur_best['time_us']:.1f} us)" if heur_best else " (N/A)"
        print(
            f"  N={N} {M:>6}x{K:<6} {label:<22} "
            f"best: yt={best['ytile']} ur={best['unrl']} "
            f"({best['time_us']:.1f} us) "
            f"heur: yt={heur_yt} ur={heur_ur}{heur_str}{marker}"
        )

    print()
    if total_best_time > 0:
        overall_regret = (total_heur_time - total_best_time) / total_best_time * 100
        print(
            f"Aggregate heuristic regret: {overall_regret:.1f}% "
            f"(sum best: {total_best_time:.0f} us, sum heur: {total_heur_time:.0f} us)"
        )

    if regressions:
        print()
        print(f"{len(regressions)} shapes where heuristic is >5% slower than best:")
        for sk, regret, best, heur in sorted(regressions, key=lambda x: -x[1]):
            M, K, N, label = sk
            print(
                f"  N={N} {M:>6}x{K:<6} {label:<22} "
                f"best yt={best['ytile']} ur={best['unrl']} ({best['time_us']:.1f} us) "
                f"vs heur yt={heur['ytile']}"
                f" ur={heur['unrl']} ({heur['time_us']:.1f} us) "
                f"+{regret:.0f}%"
            )
    print()

    # YTILE win analysis
    print("=" * 110)
    print("YTILE WIN COUNT (how often each YTILE value is part of the best config)")
    print("=" * 110)
    yt_wins = collections.Counter()
    for r in best_per_shape:
        yt_wins[r["ytile"]] += 1
    for yt in YTILES:
        print(f"  YTILE={yt}: {yt_wins.get(yt, 0)} wins")

    print()
    print("=" * 110)
    print("UNRL WIN COUNT (how often each UNRL value is part of the best config)")
    print("=" * 110)
    ur_wins = collections.Counter()
    for r in best_per_shape:
        ur_wins[r["unrl"]] += 1
    for ur in UNRLS:
        print(f"  UNRL={ur}: {ur_wins.get(ur, 0)} wins")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep fp16/bf16 skinny GEMM tuning parameters"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Batch sizes (N) to sweep (default: 1 4)",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=None,
        help="Shapes as MxK (default: all built-in shapes)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Data type (default: float16)",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    shapes = args.shapes if args.shapes else SHAPES
    run_sweep(shapes, args.batch_sizes, dtype, args.warmup, args.rep)


if __name__ == "__main__":
    main()
