#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark wvSplitK / wvSplitK_int8 / wvSplitK_int4 skinny GEMM kernels.

Compares the ROCm HIP skinny GEMM kernels against torch.nn.functional.linear
for small batch sizes (N=1..4) typical of decode.

Usage:
    python benchmark_skinny_gemm.py
    python benchmark_skinny_gemm.py --batch-sizes 1 2 3 4
    python benchmark_skinny_gemm.py --batch-sizes 1 --dtype bfloat16
    python benchmark_skinny_gemm.py --int8 --batch-sizes 1
    python benchmark_skinny_gemm.py --int4 --batch-sizes 1 2 3 4
    python benchmark_skinny_gemm.py --int4 --format-b --batch-sizes 1
"""

import argparse

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units as get_cu_count

# (M, K) shapes from real model layers.
# M = output features (weight rows), K = input features (weight cols).
# Naming follows the convention: weight is [M, K], activation is [N, K].
SHAPES = [
    # Qwen2.5-0.5B-Instruct (hidden=896, intermediate=4864)
    (9728, 896, "Qwen0.5B gate_up"),
    # Gemma-2B (hidden=2048, intermediate=16384)
    (2048, 2048, "Gemma2B q/o"),
    (2560, 2048, "Gemma2B qkv"),
    (32768, 2048, "Gemma2B gate_up"),
    (2048, 16384, "Gemma2B down"),
    # Qwen3-1.7B / Qwen3-VL-2B (hidden=2048, intermediate=6144)
    (4096, 2048, "Qwen1.7B qkv"),
    (12288, 2048, "Qwen1.7B gate_up"),
    (2048, 6144, "Qwen1.7B down"),
    # Qwen3-4B (hidden=2560, intermediate=9728)
    (2560, 2560, "Qwen3-4B q/o"),
    (6144, 2560, "Qwen3-4B qkv"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    # Qwen2.5-VL-3B (hidden=2048, intermediate=11008)
    (22016, 2048, "Qwen2.5VL-3B gate_up"),
    (2048, 11008, "Qwen2.5VL-3B down"),
    # Qwen2.5-7B / Qwen2.5-VL-7B (hidden=3584, intermediate=18944)
    (3584, 3584, "Qwen7B q/o"),
    (4608, 3584, "Qwen7B qkv"),
    (37888, 3584, "Qwen7B gate_up"),
    (3584, 18944, "Qwen7B down"),
    (152064, 3584, "Qwen7B lm_head"),
    # LLaMA-3.1-8B (hidden=4096, intermediate=14336)
    (4096, 4096, "LLaMA8B q/o"),
    (6144, 4096, "LLaMA8B qkv"),
    (28672, 4096, "LLaMA8B gate_up"),
    (4096, 14336, "LLaMA8B down"),
    # LLaMA2-7B (hidden=4096, intermediate=11008)
    (11008, 4096, "LLaMA2-7B up/gate"),
    (22016, 4096, "LLaMA2-7B gate_up"),
    (4096, 11008, "LLaMA2-7B down"),
]

LDS_SIZE_HALF = 64 * 1024 // 2  # 32K elements for non-gfx950


def fits_sml(K, N):
    """Check if K*N fits in LDS for the sml variant."""
    return K * N <= LDS_SIZE_HALF


def fits_medium(K, N):
    """Check if K*N fits within the 1.2x LDS threshold (medium/hf variant)."""
    return int(LDS_SIZE_HALF * 1.2) >= K * N


def int4_kernel_variant(M, K, N):
    """Determine which int4 kernel variant wvSplitK_int4 dispatches to."""
    prod = K * N
    if prod <= LDS_SIZE_HALF:
        return "sml"
    elif prod <= int(LDS_SIZE_HALF * 1.2):
        return "hf"
    else:
        return "N/A"


def kernel_variant(M, K, N, cu_count):
    """Determine which kernel variant wvSplitK dispatches to on gfx11.

    Tuned for RDNA 3.5 with CuCount = num_CUs (40 on Strix Halo).
    """
    prod = K * N
    sYT = (M + cu_count * 4 - 1) // (cu_count * 4)

    if sYT <= 1:
        ytile, unrl = 1, 4
    elif sYT <= 13:
        ytile, unrl = (1, 4) if K <= 2048 else (1, 1)
    elif K <= 2048:
        if sYT <= 16:
            ytile, unrl = 1, 2
        elif sYT <= 26:
            ytile, unrl = 1, 4
        else:
            ytile, unrl = 1, 1
    elif K <= 3584:
        if sYT >= 237:
            ytile, unrl = 4, 1
        elif sYT <= 39 and M % 3 == 0:
            ytile, unrl = 3, 2
        else:
            ytile, unrl = 1, 2
    else:
        # K >= 4096
        if sYT >= 237:
            ytile, unrl = 4, 1
        else:
            ytile, unrl = 1, 1

    if prod <= LDS_SIZE_HALF and M % ytile == 0:
        variant = "sml"
    elif prod <= int(LDS_SIZE_HALF * 1.2):
        variant = "hf"
    else:
        variant = "big"
    return variant, ytile, unrl


def calculate_bytes_fp16(M, K, N):
    """Total bytes for fp16 kernel: weight [M,K]*2 + act [N,K]*2 + out [N,M]*2."""
    return (M * K + N * K + N * M) * 2


def calculate_bytes_int8(M, K, N):
    """Total bytes: W[M,K]*1 + A[N,K]*2 + S[M]*2 + O[N,M]*2."""
    return M * K * 1 + N * K * 2 + M * 2 + N * M * 2


def calculate_bytes_int4(M, K, N):
    """Total bytes: W[M,K/2] + A[N,K]*2 + S[M]*2 + O[N,M]*2."""
    return M * K // 2 + N * K * 2 + M * 2 + N * M * 2


def pack_int4_format_a(values_int8, dtype=torch.float16):
    """Pack signed int4 values (in [-8,7] stored as int8) into byte-packed int8 tensor.

    For fp16: uses ExLlama shuffle for fast bitwise dequant.
    For bf16: uses sequential byte packing (low_nibble | high_nibble<<4).
    Input:  [M, K] int8 with values in [-8, 7]
    Output: [M, K/2] int8 (packed)
    """
    M, K = values_int8.shape
    if dtype == torch.float16:
        unsigned = (values_int8.to(torch.int16) + 8).to(torch.uint8)
        g = unsigned.view(M, K // 8, 8).to(torch.int32)
        shuffled = (
            g[:, :, 0]
            | (g[:, :, 2] << 4)
            | (g[:, :, 4] << 8)
            | (g[:, :, 6] << 12)
            | (g[:, :, 1] << 16)
            | (g[:, :, 3] << 20)
            | (g[:, :, 5] << 24)
            | (g[:, :, 7] << 28)
        )
        return shuffled.contiguous().view(torch.int8).contiguous()
    else:
        assert K % 2 == 0
        low = values_int8[:, 0::2] & 0xF
        high = values_int8[:, 1::2] & 0xF
        packed = (low | (high << 4)).to(torch.uint8)
        return packed.view(torch.int8).contiguous()


def pack_int4_format_b(values_int8):
    """Pack signed int4 values into int32 tensor (8 values per int32).

    Layout: bits [3:0]=k+0, [7:4]=k+1, ..., [31:28]=k+7.
    Input:  [M, K] int8 with values in [-8, 7]
    Output: [M, K/8] int32 (packed)
    """
    M, K = values_int8.shape
    assert K % 8 == 0
    reshaped = (values_int8.view(M, K // 8, 8) & 0xF).to(torch.int32)
    shifts = torch.arange(8, device=values_int8.device, dtype=torch.int32) * 4
    packed = (reshaped << shifts).sum(dim=-1).to(torch.int32)
    return packed.contiguous()


def make_benchmark(batch_size, dtype, warmup, rep):
    cu_count = get_cu_count()
    shape_labels = [f"{M}x{K}" for M, K, _ in SHAPES]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape_idx"],
            x_vals=list(range(len(SHAPES))),
            x_log=False,
            line_arg="provider",
            line_vals=["wvSplitK", "torch_linear"],
            line_names=["wvSplitK", "torch.linear"],
            styles=[("blue", "-"), ("green", "--")],
            ylabel="GiB/s",
            plot_name=f"skinny-gemm-N{batch_size}-{dtype}",
            args={},
        )
    )
    def bench_fn(shape_idx, provider):
        M, K, _label = SHAPES[shape_idx]
        N = batch_size

        weight = (torch.rand(M, K, dtype=dtype, device="cuda") - 0.5) * 0.01
        activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

        total_bytes = calculate_bytes_fp16(M, K, N)
        GIB = 1024**3

        if provider == "wvSplitK":
            fn = lambda: ops.wvSplitK(  # noqa: E731
                weight, activation.view(-1, activation.size(-1)), cu_count
            )
        else:
            fn = lambda: torch.nn.functional.linear(activation, weight)  # noqa: E731

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
        return total_bytes / (ms * 1e-3) / GIB

    return bench_fn, shape_labels


def run_table(batch_sizes, dtype, warmup, rep):
    """Run all shapes and print a combined results table."""
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    GIB = 1024**3

    print(f"GPU: {gpu_name}")
    print(f"CU count: {cu_count}")
    print(f"dtype: {dtype}")
    print(f"warmup: {warmup}, rep: {rep}")
    print()

    header = (
        f"{'N':>3} | {'M':>6} x {'K':<6} | {'Layer':<22}"
        f" | {'Variant':<5} | {'YT':>2},{' U':>2}"
        f" | {'wvSplitK':>10} | {'linear':>10} | {'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 8 != 0:
                continue

            total_bytes = calculate_bytes_fp16(M, K, N)
            weight = (torch.rand(M, K, dtype=dtype, device="cuda") - 0.5) * 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            variant, ytile, unrl = kernel_variant(M, K, N, cu_count)

            fn_sk = lambda w=weight, a=activation: ops.wvSplitK(  # noqa: E731
                w, a.view(-1, a.size(-1)), cu_count
            )
            fn_lin = lambda w=weight, a=activation: torch.nn.functional.linear(  # noqa: E731
                a, w
            )

            ms_sk = triton.testing.do_bench(
                fn_sk, warmup=warmup, rep=rep, return_mode="median"
            )
            ms_lin = triton.testing.do_bench(
                fn_lin, warmup=warmup, rep=rep, return_mode="median"
            )

            bw_sk = total_bytes / (ms_sk * 1e-3) / GIB
            bw_lin = total_bytes / (ms_lin * 1e-3) / GIB
            speedup = ms_lin / ms_sk

            print(
                f"{N:>3} | {M:>6} x {K:<6} | {label:<22}"
                f" | {variant:<5} | {ytile:>2}, {unrl:>2}"
                f" | {bw_sk:>8.1f}  | {bw_lin:>8.1f}  | {speedup:>7.2f}x"
            )
        if batch_sizes[-1] != N:
            print()


def run_int8_accuracy(dtype, batch_sizes):
    """Validate int8 kernel accuracy against dequant + torch.linear reference."""
    cu_count = get_cu_count()
    print("=" * 80)
    print("INT8 ACCURACY VALIDATION")
    print("=" * 80)
    print(f"dtype: {dtype}")
    print()

    all_pass = True
    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_sml(K, N):
                continue

            weight_int8 = torch.randint(
                -128, 127, (M, K), dtype=torch.int8, device="cuda"
            )
            scale = torch.rand(M, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            result = ops.wvSplitK_int8(
                weight_int8, activation.view(-1, activation.size(-1)), scale, cu_count
            )

            weight_dequant = weight_int8.to(dtype) * scale[:, None]
            reference = torch.nn.functional.linear(activation, weight_dequant)

            max_err = (result - reference).abs().max().item()
            mean_err = (result - reference).abs().mean().item()

            atol = 1.0 if dtype == torch.bfloat16 else 0.5
            passed = max_err < atol
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            print(
                f"  [{status}] N={N} {M:>6}x{K:<6} {label:<22}"
                f"  max_err={max_err:.4f}  mean_err={mean_err:.6f}"
            )

    print()
    if all_pass:
        print("All accuracy checks PASSED.")
    else:
        print("WARNING: Some accuracy checks FAILED!")
    print()
    return all_pass


def run_int8_comparison(dtype, batch_sizes, warmup, rep):
    """Run int8 vs fp16 bandwidth comparison and produce a report."""
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    GIB = 1024**3

    print("=" * 100)
    print(
        "INT8 vs FP16 SKINNY GEMM BANDWIDTH COMPARISON"
        f" (N={','.join(map(str, batch_sizes))})"
    )
    print("=" * 100)
    print(f"GPU: {gpu_name}")
    print(f"CU count: {cu_count}")
    print(f"dtype: {dtype}")
    print(f"warmup: {warmup}, rep: {rep}")
    print()

    header = (
        f"{'N':>2} | {'M':>6} x {'K':<6} | {'Layer':<22}"
        f" | {'fp16 us':>8} | {'fp16 GiB/s':>10}"
        f" | {'int8 us':>8} | {'int8 GiB/s':>10}"
        f" | {'Speedup':>8} | {'BW ratio':>9}"
    )
    lines = []
    lines.append(header)
    lines.append("-" * len(header))

    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_sml(K, N):
                continue

            weight_fp16 = (torch.rand(M, K, dtype=dtype, device="cuda") - 0.5) * 0.01
            weight_int8 = torch.randint(
                -128, 127, (M, K), dtype=torch.int8, device="cuda"
            )
            scale = torch.rand(M, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            fn_fp16 = lambda w=weight_fp16, a=activation: ops.wvSplitK(  # noqa: E731
                w, a.view(-1, a.size(-1)), cu_count
            )
            fn_int8 = lambda wi=weight_int8, a=activation, s=scale: ops.wvSplitK_int8(  # noqa: E731
                wi, a.view(-1, a.size(-1)), s, cu_count
            )

            ms_fp16 = triton.testing.do_bench(
                fn_fp16, warmup=warmup, rep=rep, return_mode="median"
            )
            ms_int8 = triton.testing.do_bench(
                fn_int8, warmup=warmup, rep=rep, return_mode="median"
            )

            bytes_fp16 = calculate_bytes_fp16(M, K, N)
            bytes_int8 = calculate_bytes_int8(M, K, N)

            bw_fp16 = bytes_fp16 / (ms_fp16 * 1e-3) / GIB
            bw_int8 = bytes_int8 / (ms_int8 * 1e-3) / GIB
            speedup = ms_fp16 / ms_int8
            bw_ratio = bw_int8 / bw_fp16 if bw_fp16 > 0 else 0

            line = (
                f"{N:>2} | {M:>6} x {K:<6} | {label:<22}"
                f" | {ms_fp16 * 1000:>8.1f} | {bw_fp16:>10.1f}"
                f" | {ms_int8 * 1000:>8.1f} | {bw_int8:>10.1f}"
                f" | {speedup:>7.2f}x | {bw_ratio:>8.2f}x"
            )
            lines.append(line)

        if batch_sizes[-1] != N:
            lines.append("")

    report = "\n".join(lines)
    print(report)
    print()

    report_path = "/scratch/mgehre/tmp/int8_vs_fp16_skinny_gemm.txt"
    full_report = (
        "INT8 vs FP16 SKINNY GEMM BANDWIDTH COMPARISON"
        f" (N={','.join(map(str, batch_sizes))})\n"
        f"GPU: {gpu_name}\n"
        f"CU count: {cu_count}\n"
        f"dtype: {dtype}\n"
        f"warmup: {warmup}, rep: {rep}\n\n" + report + "\n"
    )
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"Report saved to {report_path}")


def run_int4_accuracy(dtype, batch_sizes, use_format_b=False):
    """Validate int4 kernel accuracy against dequant + torch.linear reference."""
    cu_count = get_cu_count()
    fmt_name = "Format B (int32)" if use_format_b else "Format A (int8)"
    print("=" * 80)
    print(f"INT4 ACCURACY VALIDATION ({fmt_name})")
    print("=" * 80)
    print(f"dtype: {dtype}")
    print()

    all_pass = True
    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_medium(K, N):
                continue

            variant = int4_kernel_variant(M, K, N)

            values_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
            scale = torch.rand(M, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            if use_format_b:
                weight_packed = pack_int4_format_b(values_int4)
            else:
                weight_packed = pack_int4_format_a(values_int4, dtype)

            result = ops.wvSplitK_int4(
                weight_packed, activation.view(-1, activation.size(-1)), scale, cu_count
            )

            weight_dequant = values_int4.to(dtype) * scale[:, None]
            reference = torch.nn.functional.linear(activation, weight_dequant)

            max_err = (result - reference).abs().max().item()
            mean_err = (result - reference).abs().mean().item()

            atol = 1.0 if dtype == torch.bfloat16 else 0.5
            passed = max_err < atol
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            print(
                f"  [{status}] N={N} {M:>6}x{K:<6} {label:<22}"
                f"  [{variant:>3}]  max_err={max_err:.4f}  mean_err={mean_err:.6f}"
            )

    print()
    if all_pass:
        print("All accuracy checks PASSED.")
    else:
        print("WARNING: Some accuracy checks FAILED!")
    print()
    return all_pass


def run_int4_comparison(dtype, batch_sizes, warmup, rep, use_format_b=False):
    """Run int4 vs fp16 (and int8) bandwidth comparison."""
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    GIB = 1024**3
    fmt_name = "Format B (int32)" if use_format_b else "Format A (int8)"

    print("=" * 130)
    print(
        "INT4 vs FP16 vs INT8 SKINNY GEMM COMPARISON"
        f" (N={','.join(map(str, batch_sizes))})"
        f" [{fmt_name}]"
    )
    print("=" * 130)
    print(f"GPU: {gpu_name}")
    print(f"CU count: {cu_count}")
    print(f"dtype: {dtype}")
    print(f"warmup: {warmup}, rep: {rep}")
    print()

    header = (
        f"{'N':>2} | {'M':>6} x {'K':<6} | {'Layer':<22} | {'Var':<3}"
        f" | {'fp16 us':>8} | {'fp16 GiB/s':>10}"
        f" | {'int8 us':>8} | {'int8 GiB/s':>10}"
        f" | {'int4 us':>8} | {'int4 GiB/s':>10}"
        f" | {'i4/fp16':>7} | {'i4/i8':>6}"
    )
    lines = []
    lines.append(header)
    lines.append("-" * len(header))

    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_medium(K, N):
                continue

            variant = int4_kernel_variant(M, K, N)

            weight_fp16 = (torch.rand(M, K, dtype=dtype, device="cuda") - 0.5) * 0.01
            values_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
            scale = torch.rand(M, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            if use_format_b:
                weight_int4_packed = pack_int4_format_b(values_int4)
            else:
                weight_int4_packed = pack_int4_format_a(values_int4, dtype)

            fn_fp16 = lambda w=weight_fp16, a=activation: ops.wvSplitK(  # noqa: E731
                w, a.view(-1, a.size(-1)), cu_count
            )
            fn_int4 = (
                lambda wi4=weight_int4_packed, a=activation, s=scale: ops.wvSplitK_int4(  # noqa: E731
                    wi4, a.view(-1, a.size(-1)), s, cu_count
                )
            )

            ms_fp16 = triton.testing.do_bench(
                fn_fp16, warmup=warmup, rep=rep, return_mode="median"
            )
            ms_int4 = triton.testing.do_bench(
                fn_int4, warmup=warmup, rep=rep, return_mode="median"
            )

            bytes_fp16 = calculate_bytes_fp16(M, K, N)
            bytes_int4 = calculate_bytes_int4(M, K, N)

            bw_fp16 = bytes_fp16 / (ms_fp16 * 1e-3) / GIB
            bw_int4 = bytes_int4 / (ms_int4 * 1e-3) / GIB
            speedup_vs_fp16 = ms_fp16 / ms_int4

            # int8 only available for sml shapes
            if fits_sml(K, N):
                weight_int8 = torch.randint(
                    -128, 127, (M, K), dtype=torch.int8, device="cuda"
                )
                fn_int8 = (
                    lambda wi=weight_int8, a=activation, s=scale: ops.wvSplitK_int8(  # noqa: E731
                        wi, a.view(-1, a.size(-1)), s, cu_count
                    )
                )
                ms_int8 = triton.testing.do_bench(
                    fn_int8, warmup=warmup, rep=rep, return_mode="median"
                )
                bytes_int8 = calculate_bytes_int8(M, K, N)
                bw_int8 = bytes_int8 / (ms_int8 * 1e-3) / GIB
                speedup_vs_int8 = ms_int8 / ms_int4
                int8_us_str = f"{ms_int8 * 1000:>8.1f}"
                int8_bw_str = f"{bw_int8:>10.1f}"
                int8_sp_str = f"{speedup_vs_int8:>5.2f}x"
            else:
                int8_us_str = f"{'N/A':>8}"
                int8_bw_str = f"{'N/A':>10}"
                int8_sp_str = f"{'N/A':>6}"

            line = (
                f"{N:>2} | {M:>6} x {K:<6} | {label:<22} | {variant:<3}"
                f" | {ms_fp16 * 1000:>8.1f} | {bw_fp16:>10.1f}"
                f" | {int8_us_str} | {int8_bw_str}"
                f" | {ms_int4 * 1000:>8.1f} | {bw_int4:>10.1f}"
                f" | {speedup_vs_fp16:>6.2f}x | {int8_sp_str}"
            )
            lines.append(line)

        if batch_sizes[-1] != N:
            lines.append("")

    report = "\n".join(lines)
    print(report)
    print()

    fmt_suffix = "_formatB" if use_format_b else "_formatA"
    report_path = f"/scratch/mgehre/tmp/int4_vs_fp16_skinny_gemm{fmt_suffix}.txt"
    full_report = (
        "INT4 vs FP16 vs INT8 SKINNY GEMM COMPARISON"
        f" (N={','.join(map(str, batch_sizes))})"
        f" [{fmt_name}]\n"
        f"GPU: {gpu_name}\n"
        f"CU count: {cu_count}\n"
        f"dtype: {dtype}\n"
        f"warmup: {warmup}, rep: {rep}\n\n" + report + "\n"
    )
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"Report saved to {report_path}")


def run_int4_group_accuracy(dtype, batch_sizes, group_size):
    """Validate int4 grouped kernel accuracy against dequant + torch.linear."""
    cu_count = get_cu_count()
    print("=" * 80)
    print(f"INT4 GROUPED ACCURACY VALIDATION (group_size={group_size})")
    print("=" * 80)
    print(f"dtype: {dtype}")
    print()

    all_pass = True
    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_medium(K, N):
                continue
            if K % group_size != 0:
                continue

            variant = int4_kernel_variant(M, K, N)
            num_groups = K // group_size
            values_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
            scale = torch.rand(M, num_groups, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            weight_packed = pack_int4_format_a(values_int4, dtype)

            result = ops.wvSplitK_int4_g(
                weight_packed,
                activation.view(-1, activation.size(-1)),
                scale,
                cu_count,
                group_size,
            )

            weight_dequant = values_int4.to(dtype).view(M, num_groups, group_size)
            weight_dequant = (weight_dequant * scale.unsqueeze(-1)).view(M, K)
            reference = torch.nn.functional.linear(activation, weight_dequant)

            max_err = (result - reference).abs().max().item()
            mean_err = (result - reference).abs().mean().item()

            atol = 1.0 if dtype == torch.bfloat16 else 0.5
            passed = max_err < atol
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            print(
                f"  [{status}] N={N} {M:>6}x{K:<6} {label:<22}"
                f"  [{variant:>3}]  max_err={max_err:.4f}  mean_err={mean_err:.6f}"
            )

    print()
    if all_pass:
        print("All accuracy checks PASSED.")
    else:
        print("WARNING: Some accuracy checks FAILED!")
    print()
    return all_pass


def calculate_bytes_int4_group(M, K, N, group_size):
    """Bytes for int4 grouped: weight + act + group_scales + output."""
    num_groups = K // group_size
    return M * K // 2 + N * K * 2 + M * num_groups * 2 + N * M * 2


def run_int4_group_table(batch_sizes, dtype, warmup, rep, group_size):
    """Run all shapes for int4 grouped and print a combined results table."""
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    GIB = 1024**3

    print(f"GPU: {gpu_name}")
    print(f"CU count: {cu_count}")
    print(f"dtype: {dtype}")
    print(f"int4 group_size: {group_size}")
    print(f"warmup: {warmup}, rep: {rep}")
    print()

    header = (
        f"{'N':>3} | {'M':>6} x {'K':<6} | {'Layer':<22}"
        f" | {'Variant':<5}"
        f" | {'wvSplitK_i4g':>12} | {'linear':>10} | {'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for N in batch_sizes:
        for M, K, label in SHAPES:
            if K % 16 != 0 or not fits_medium(K, N):
                continue
            if K % group_size != 0:
                continue

            variant = int4_kernel_variant(M, K, N)
            num_groups = K // group_size
            total_bytes = calculate_bytes_int4_group(M, K, N, group_size)

            values_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
            scale = torch.rand(M, num_groups, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01
            weight_packed = pack_int4_format_a(values_int4, dtype)

            weight_dequant = values_int4.to(dtype).view(M, num_groups, group_size)
            weight_dequant = (weight_dequant * scale.unsqueeze(-1)).view(M, K)

            fn_sk = (
                lambda wi=weight_packed,
                a=activation,
                s=scale,
                gs=group_size: ops.wvSplitK_int4_g(  # noqa: E731
                    wi, a.view(-1, a.size(-1)), s, cu_count, gs
                )
            )
            fn_lin = lambda w=weight_dequant, a=activation: torch.nn.functional.linear(  # noqa: E731
                a, w
            )

            ms_sk = triton.testing.do_bench(
                fn_sk, warmup=warmup, rep=rep, return_mode="median"
            )
            ms_lin = triton.testing.do_bench(
                fn_lin, warmup=warmup, rep=rep, return_mode="median"
            )

            bw_sk = total_bytes / (ms_sk * 1e-3) / GIB
            bw_lin = total_bytes / (ms_lin * 1e-3) / GIB
            speedup = ms_lin / ms_sk

            print(
                f"{N:>3} | {M:>6} x {K:<6} | {label:<22}"
                f" | {variant:<5}"
                f" | {bw_sk:>10.1f}  | {bw_lin:>10.1f}  | {speedup:>7.2f}x"
            )
        if batch_sizes[-1] != N:
            print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark wvSplitK skinny GEMM")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Batch sizes (N) to benchmark (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Data type (default: float16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Warmup iterations (default: 50)"
    )
    parser.add_argument(
        "--rep", type=int, default=200, help="Benchmark repetitions (default: 200)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate triton perf_report plots (one per batch size)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Benchmark int8 kernel: accuracy then int8 vs fp16",
    )
    parser.add_argument(
        "--int4",
        action="store_true",
        help="Benchmark int4 kernel: accuracy then int4 vs fp16/int8",
    )
    parser.add_argument(
        "--format-b",
        action="store_true",
        help="Use Format B (int32 packing) for int4 instead of Format A (int8 packing)",
    )
    parser.add_argument(
        "--int4-group",
        type=int,
        default=0,
        metavar="GROUP_SIZE",
        help="Benchmark int4 grouped kernel with given group size (32 or 128)",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.int4_group > 0:
        run_int4_group_accuracy(dtype, args.batch_sizes, args.int4_group)
        run_int4_group_table(
            args.batch_sizes, dtype, args.warmup, args.rep, args.int4_group
        )
    elif args.int4:
        run_int4_accuracy(dtype, args.batch_sizes, use_format_b=args.format_b)
        run_int4_comparison(
            dtype, args.batch_sizes, args.warmup, args.rep, use_format_b=args.format_b
        )
    elif args.int8:
        run_int8_accuracy(dtype, args.batch_sizes)
        run_int8_comparison(dtype, args.batch_sizes, args.warmup, args.rep)
    elif args.plot:
        for N in args.batch_sizes:
            bench_fn, shape_labels = make_benchmark(N, dtype, args.warmup, args.rep)
            bench_fn.run(
                print_data=True,
                show_plots=False,
                save_path=f"/scratch/mgehre/tmp/skinny_gemm_N{N}",
            )
    else:
        run_table(args.batch_sizes, dtype, args.warmup, args.rep)


if __name__ == "__main__":
    main()
