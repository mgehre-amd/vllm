#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AWQ Triton kernel that matches the HIP kernel approach.

HIP kernel strategy:
- OUTPUT_PER_THREAD = 8 (each thread computes 8 output elements)
- SPLIT_K parallelism (divide K across multiple thread groups)
- PIPELINE_DEPTH = 16 (prefetch 16 rows of weights)
- 32 threads per split, each processing 8 outputs -> 256 outputs per split
- fp32 accumulators for precision
- Shared memory reduction across splits

Key insight: The HIP kernel is K-major (iterates K in outer loop, outputs per thread in inner),
while most Triton AWQ kernels are N-major (iterate N in outer, K in inner).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def awq_gemv_hip_style_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [SPLIT_K, N] fp16 for partial results
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    THREADS_PER_SPLIT: tl.constexpr = 32,
    OUTPUT_PER_THREAD: tl.constexpr = 8,
):
    """
    AWQ GEMV kernel matching HIP kernel approach.

    Each block processes THREADS_PER_SPLIT * OUTPUT_PER_THREAD = 256 outputs.
    With SPLIT_K splits, each split processes K/SPLIT_K elements.
    """
    # Thread and block indices
    pid_n = tl.program_id(0)  # Which block of N
    pid_k = tl.program_id(1)  # Which K split

    # Calculate output column range for this thread
    # Each block handles THREADS_PER_SPLIT * OUTPUT_PER_THREAD outputs
    OUTPUTS_PER_BLOCK: tl.constexpr = THREADS_PER_SPLIT * OUTPUT_PER_THREAD

    # Thread ID within the block (0 to THREADS_PER_SPLIT-1)
    # In Triton, we simulate this by having each "program" be like one HIP thread
    # But Triton launches program_id(0) * program_id(1) programs

    # Actually in Triton we need a different approach - each program handles
    # a portion of the work. Let's have each program handle OUTPUT_PER_THREAD outputs
    # and use split-K

    # Column start for this program
    col_start = pid_n * OUTPUT_PER_THREAD

    # Check bounds
    if col_start >= N:
        return

    # Column indices [OUTPUT_PER_THREAD]
    col_offs = col_start + tl.arange(0, OUTPUT_PER_THREAD)

    # Packed column indices for qweight/qzeros
    col_packed = col_offs // 8
    col_in_pack = col_offs % 8

    # AWQ unpacking shifts: [0,4,1,5,2,6,3,7] order
    # For position i in unpacked: shift = (i//2)*4 + (i%2)*16
    shifts = (col_in_pack // 2) * 4 + (col_in_pack % 2) * 16

    # Calculate K range for this split
    num_groups = K // GROUP_SIZE
    groups_per_split = num_groups // SPLIT_K
    g_start = pid_k * groups_per_split
    g_end = g_start + groups_per_split
    k_start = g_start * GROUP_SIZE

    N_packed = N // 8

    # Accumulators in fp32 [OUTPUT_PER_THREAD]
    acc = tl.zeros([OUTPUT_PER_THREAD], dtype=tl.float32)

    # Main loop over groups assigned to this split
    for g in range(g_start, g_end):
        # Load scales [OUTPUT_PER_THREAD]
        scale_ptrs = scales_ptr + g * N + col_offs
        scales = tl.load(scale_ptrs).to(tl.float16)

        # Load and unpack zeros [OUTPUT_PER_THREAD]
        qz_ptrs = qzeros_ptr + g * N_packed + col_packed
        qz = tl.load(qz_ptrs)
        zeros = ((qz >> shifts) & 0xF).to(tl.float16)

        # Precompute bias = -zeros * scales for efficiency
        bias = -zeros * scales

        # Inner loop over K elements in this group
        k_base = g * GROUP_SIZE
        for k_off in range(GROUP_SIZE):
            k = k_base + k_off

            # Load activation scalar
            x = tl.load(input_ptr + k).to(tl.float16)

            # Load weights and unpack [OUTPUT_PER_THREAD]
            qw_ptrs = qweight_ptr + k * N_packed + col_packed
            qw = tl.load(qw_ptrs)
            w = ((qw >> shifts) & 0xF).to(tl.float16)

            # Dequantize: (w - zeros) * scales = w * scales + bias
            dequant = w * scales + bias

            # Accumulate in fp32
            acc += (x * dequant).to(tl.float32)

    # Store partial results
    out_offs = pid_k * N + col_offs
    tl.store(output_ptr + out_offs, acc.to(tl.float16))


@triton.jit
def reduce_partial_kernel(
    partial_ptr,  # [SPLIT_K, N] fp16
    output_ptr,  # [N] fp16
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    OUTPUT_PER_THREAD: tl.constexpr,
):
    """Reduce partial results from split-K."""
    pid = tl.program_id(0)
    col_start = pid * OUTPUT_PER_THREAD
    col_offs = col_start + tl.arange(0, OUTPUT_PER_THREAD)

    # Accumulate in fp32
    acc = tl.zeros([OUTPUT_PER_THREAD], dtype=tl.float32)
    for k_split in range(SPLIT_K):
        partial = tl.load(partial_ptr + k_split * N + col_offs).to(tl.float32)
        acc += partial

    # Store final result
    tl.store(output_ptr + col_offs, acc.to(tl.float16))


@triton.jit
def awq_gemv_v3_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [SPLIT_K, N] fp16 for partial results
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    AWQ GEMV kernel v3 - matching HIP thread structure exactly.

    Each program handles exactly 8 outputs (one uint32 load per K row).
    This matches HIP's OUTPUT_PER_THREAD=8 exactly.
    """
    OUTPUT_PER_THREAD: tl.constexpr = 8

    pid_n = tl.program_id(0)  # Which group of 8 outputs
    pid_k = tl.program_id(1)  # Which K split

    # Column range for this program - exactly 8 outputs
    col_start = pid_n * OUTPUT_PER_THREAD
    col_offs = col_start + tl.arange(0, OUTPUT_PER_THREAD)

    # For 8 consecutive columns, they all map to the same packed uint32
    # col_packed = col_start // 8 (single value, same for all 8)
    packed_col = col_start // 8

    # Position within the packed uint32: 0,1,2,3,4,5,6,7
    col_in_pack = tl.arange(0, OUTPUT_PER_THREAD)

    # AWQ unpacking shifts for positions 0-7
    shifts = (col_in_pack // 2) * 4 + (col_in_pack % 2) * 16

    # Constexpr values
    N_PACKED: tl.constexpr = N // 8

    # K range for this split
    num_groups = K // GROUP_SIZE
    groups_per_split = num_groups // SPLIT_K
    g_start = pid_k * groups_per_split
    g_end = g_start + groups_per_split

    # fp32 accumulators [8]
    acc = tl.zeros([OUTPUT_PER_THREAD], dtype=tl.float32)

    # Process groups
    for g in tl.range(g_start, g_end, flatten=True):
        # Load scales [8] - consecutive, so one vector load
        scale_ptr = scales_ptr + g * N + col_start
        scales = tl.load(scale_ptr + tl.arange(0, OUTPUT_PER_THREAD)).to(tl.float16)

        # Load one packed uint32 for zeros and unpack to [8]
        qz_ptr = qzeros_ptr + g * N_PACKED + packed_col
        qz = tl.load(qz_ptr)  # Single uint32
        # Broadcast and shift to extract all 8 values
        qz_broadcast = tl.full([OUTPUT_PER_THREAD], qz, dtype=tl.int32)
        zeros = ((qz_broadcast >> shifts) & 0xF).to(tl.float16)

        k_base = g * GROUP_SIZE

        # Process K elements in this group
        for k in tl.range(k_base, k_base + GROUP_SIZE, flatten=True):
            # Load activation scalar
            x = tl.load(input_ptr + k).to(tl.float16)

            # Load one packed uint32 for weights and unpack to [8]
            qw_ptr = qweight_ptr + k * N_PACKED + packed_col
            qw = tl.load(qw_ptr)  # Single uint32
            qw_broadcast = tl.full([OUTPUT_PER_THREAD], qw, dtype=tl.int32)
            w = ((qw_broadcast >> shifts) & 0xF).to(tl.float16)

            # Dequantize: (w - zeros) * scales
            dequant = (w - zeros) * scales

            # Accumulate in fp32
            acc += (x * dequant).to(tl.float32)

    # Store partial results
    out_offs = pid_k * N + col_offs
    tl.store(output_ptr + out_offs, acc.to(tl.float16))


@triton.jit
def awq_gemv_simple_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [N] fp16
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr = 8,  # Unroll factor for inner loop
):
    """
    Simple AWQ GEMV kernel with chunked K processing.
    Compute entirely in fp32 to minimize conversions.
    """
    pid = tl.program_id(0)

    # Column range for this block - BLOCK_N outputs
    col_start = pid * BLOCK_N
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Each 8 consecutive outputs share one packed int32
    col_packed = col_offs // 8
    col_in_pack = col_offs % 8

    # AWQ unpacking: bits 0-3 go to cols 0,2,4,6; bits 16-19 go to cols 1,3,5,7
    shifts = (col_in_pack // 2) * 4 + (col_in_pack % 2) * 16

    N_PACKED: tl.constexpr = N // 8
    num_groups = K // GROUP_SIZE
    ITERS_PER_GROUP: tl.constexpr = GROUP_SIZE // UNROLL

    # fp32 accumulators
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Precompute base pointers
    qweight_base = qweight_ptr + col_packed
    qzeros_base = qzeros_ptr + col_packed
    scales_base = scales_ptr + col_offs

    # Loop over groups - test without flatten
    for g in range(num_groups):
        # Load scales and zeros once per group (convert to fp32 once)
        scales_f32 = tl.load(scales_base + g * N).to(tl.float32)
        qz = tl.load(qzeros_base + g * N_PACKED)
        zeros_f32 = ((qz >> shifts) & 0xF).to(tl.float32)

        k_base = g * GROUP_SIZE

        # Process GROUP_SIZE elements in chunks of UNROLL
        for chunk in range(ITERS_PER_GROUP):
            k_chunk = k_base + chunk * UNROLL

            # Static unroll
            for u in tl.static_range(UNROLL):
                k = k_chunk + u

                # Load activation and convert to fp32
                x_f32 = tl.load(input_ptr + k).to(tl.float32)

                # Vectorized weight load with non-temporal hint (like HIP's __builtin_nontemporal_load)
                qw = tl.load(qweight_base + k * N_PACKED, cache_modifier=".cg")
                w_f32 = ((qw >> shifts) & 0xF).to(tl.float32)

                # Dequantize in fp32: (w - z) * s * x
                dequant = (w_f32 - zeros_f32) * scales_f32
                acc += x_f32 * dequant

    # Store output (single conversion at the end)
    tl.store(output_ptr + col_offs, acc.to(tl.float16))


def awq_gemv_simple(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    block_n: int = 256,
    num_warps: int = 4,
    unroll: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Simple AWQ GEMV - no split-K."""
    if input.dim() == 2:
        input = input.squeeze(0)
    K = input.shape[0]
    N = qweight.shape[1] * 8
    num_groups = qzeros.shape[0]
    group_size = K // num_groups

    assert group_size % unroll == 0, f"group_size={group_size} must be divisible by unroll={unroll}"

    output = torch.zeros(N, dtype=torch.float16, device=input.device)

    num_n_blocks = N // block_n
    grid = (num_n_blocks,)

    awq_gemv_simple_kernel[grid](
        input, qweight, qzeros, scales, output,
        K=K, N=N, GROUP_SIZE=group_size, BLOCK_N=block_n, UNROLL=unroll,
        num_warps=num_warps, num_stages=num_stages,
    )

    return output.unsqueeze(0)


def awq_gemv_v3(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k: int = 8,
    num_warps: int = 1,
) -> torch.Tensor:
    """AWQ GEMV v3 - matching HIP thread structure."""
    if input.dim() == 2:
        input = input.squeeze(0)
    K = input.shape[0]
    N = qweight.shape[1] * 8
    num_groups = qzeros.shape[0]
    group_size = K // num_groups

    assert num_groups % split_k == 0
    assert N % 8 == 0

    OUTPUT_PER_THREAD = 8
    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device=input.device)

    num_n_blocks = N // OUTPUT_PER_THREAD
    grid = (num_n_blocks, split_k)

    awq_gemv_v3_kernel[grid](
        input,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )

    output = torch.zeros(N, dtype=torch.float16, device=input.device)
    reduce_partial_kernel[(num_n_blocks,)](
        partial_output,
        output,
        N=N,
        SPLIT_K=split_k,
        OUTPUT_PER_THREAD=OUTPUT_PER_THREAD,
        num_warps=1,
    )

    return output.unsqueeze(0)


def awq_gemv_hip_style(
    input: torch.Tensor,  # [K] or [1, K]
    qweight: torch.Tensor,  # [K, N//8]
    qzeros: torch.Tensor,  # [K//G, N//8]
    scales: torch.Tensor,  # [K//G, N]
    split_k: int = 8,
    num_warps: int = 1,
) -> torch.Tensor:
    """
    AWQ GEMV using HIP-style approach.

    Args:
        input: Activation tensor [K] or [1, K]
        qweight: Quantized weights [K, N//8]
        qzeros: Quantized zeros [K//G, N//8]
        scales: Scales [K//G, N]
        split_k: Number of K splits
        num_warps: Number of warps per block

    Returns:
        Output tensor [N] or [1, N]
    """
    # Handle input shape
    if input.dim() == 2:
        input = input.squeeze(0)
    K = input.shape[0]
    N = qweight.shape[1] * 8
    num_groups = qzeros.shape[0]
    group_size = K // num_groups

    # Validate split_k
    assert num_groups % split_k == 0, f"num_groups={num_groups} must be divisible by split_k={split_k}"

    OUTPUT_PER_THREAD = 8
    assert N % OUTPUT_PER_THREAD == 0, f"N={N} must be divisible by {OUTPUT_PER_THREAD}"

    # Allocate partial output
    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device=input.device)

    # Launch kernel
    num_n_blocks = N // OUTPUT_PER_THREAD
    grid = (num_n_blocks, split_k)

    awq_gemv_hip_style_kernel[grid](
        input,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )

    # Reduce partial results
    output = torch.zeros(N, dtype=torch.float16, device=input.device)
    reduce_partial_kernel[(num_n_blocks,)](
        partial_output,
        output,
        N=N,
        SPLIT_K=split_k,
        OUTPUT_PER_THREAD=OUTPUT_PER_THREAD,
        num_warps=1,
    )

    return output.unsqueeze(0)


@triton.jit
def awq_gemv_hip_style_v2_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [SPLIT_K, N] fp16 for partial results
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr = 256,  # Number of outputs per block
    PIPELINE_DEPTH: tl.constexpr = 16,  # Match HIP kernel
):
    """
    AWQ GEMV kernel v2 - matching HIP kernel approach.

    Uses nested loops: groups -> iters -> pipeline_depth (unrolled)
    Optimized with pre-computed pointers and incremental addressing.
    """
    pid_n = tl.program_id(0)  # Which block of N
    pid_k = tl.program_id(1)  # Which K split

    # Column range for this block - NO MASKING
    col_start = pid_n * BLOCK_N
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Packed column indices (each uint32 holds 8 int4 values)
    col_packed = col_offs // 8
    col_in_pack = col_offs % 8

    # AWQ unpacking shifts: [0,4,1,5,2,6,3,7] order
    shifts = (col_in_pack // 2) * 4 + (col_in_pack % 2) * 16

    # Constexpr values
    N_PACKED: tl.constexpr = N // 8
    ITERS_PER_GROUP: tl.constexpr = GROUP_SIZE // PIPELINE_DEPTH

    # K range for this split
    num_groups = K // GROUP_SIZE
    groups_per_split = num_groups // SPLIT_K
    g_start = pid_k * groups_per_split
    g_end = g_start + groups_per_split

    # Pre-compute base pointer offsets (element offsets, not byte offsets)
    # These will be scaled by element size in tl.load
    k_start = g_start * GROUP_SIZE

    # Weight row stride = N_PACKED elements per row
    # W_STRIDE: tl.constexpr = N_PACKED

    # fp32 accumulators [BLOCK_N]
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Base pointers for this column block
    weight_col_offset = col_packed  # [BLOCK_N // 8] but broadcast as BLOCK_N
    scale_col_offset = col_offs     # [BLOCK_N]
    zero_col_offset = col_packed    # [BLOCK_N]

    # Process groups
    for g in tl.range(g_start, g_end, flatten=True):
        # Group offsets
        group_scale_base = g * N
        group_zero_base = g * N_PACKED

        # Load scales [BLOCK_N] - once per group (fp32 for precision)
        scales_f32 = tl.load(scales_ptr + group_scale_base + scale_col_offset).to(tl.float32)

        # Load and unpack zeros [BLOCK_N] - once per group (fp32 for consistency)
        qz = tl.load(qzeros_ptr + group_zero_base + zero_col_offset)
        zeros_f32 = ((qz >> shifts) & 0xF).to(tl.float32)

        # Base K for this group
        k_base = g * GROUP_SIZE

        # Inner loop over K with PIPELINE_DEPTH unrolling
        for iter_idx in tl.range(0, ITERS_PER_GROUP, flatten=True):
            k_iter_base = k_base + iter_idx * PIPELINE_DEPTH

            # Unroll PIPELINE_DEPTH iterations
            for p in tl.static_range(PIPELINE_DEPTH):
                k = k_iter_base + p

                # Load activation and convert to fp32
                x_f32 = tl.load(input_ptr + k).to(tl.float32)

                # Load weights with cache hint, convert to fp32
                weight_row_base = k * N_PACKED
                qw = tl.load(qweight_ptr + weight_row_base + weight_col_offset, cache_modifier=".cg")
                w_f32 = ((qw >> shifts) & 0xF).to(tl.float32)

                # Dequantize in fp32 and accumulate
                dequant = (w_f32 - zeros_f32) * scales_f32
                acc += x_f32 * dequant

    # Store partial results
    out_offs = pid_k * N + col_offs
    tl.store(output_ptr + out_offs, acc.to(tl.float16))


def awq_gemv_hip_style_v2(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k: int = 8,
    block_n: int = 256,
    num_warps: int = 4,
    num_stages: int = 0,
    pipeline_depth: int = 16,
) -> torch.Tensor:
    """AWQ GEMV using HIP-style approach v2 with pipelining."""
    if input.dim() == 2:
        input = input.squeeze(0)
    K = input.shape[0]
    N = qweight.shape[1] * 8
    num_groups = qzeros.shape[0]
    group_size = K // num_groups

    assert num_groups % split_k == 0
    assert N % block_n == 0
    assert group_size % pipeline_depth == 0, f"group_size must be divisible by PIPELINE_DEPTH={pipeline_depth}"

    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device=input.device)

    num_n_blocks = N // block_n
    grid = (num_n_blocks, split_k)

    awq_gemv_hip_style_v2_kernel[grid](
        input,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        SPLIT_K=split_k,
        BLOCK_N=block_n,
        PIPELINE_DEPTH=pipeline_depth,
        num_warps=num_warps,
        num_stages=num_stages if num_stages > 0 else 1,
    )

    output = torch.zeros(N, dtype=torch.float16, device=input.device)
    reduce_partial_kernel[(num_n_blocks,)](
        partial_output,
        output,
        N=N,
        SPLIT_K=split_k,
        OUTPUT_PER_THREAD=block_n,
        num_warps=1,
    )

    return output.unsqueeze(0)


def benchmark_kernels():
    """Benchmark the Triton HIP-style kernel against the actual HIP kernel."""
    from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton

    # Target shape: 22016 x 4096 (N x K)
    N, K = 22016, 4096
    group_size = 128
    num_groups = K // group_size

    print(f"\nBenchmarking AWQ GEMV kernels on {N}x{K} (NxK), group_size={group_size}")
    print("=" * 80)

    # Create test data
    torch.manual_seed(42)
    input_tensor = torch.randn(K, dtype=torch.float16, device="cuda")
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda")
    scales = torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01

    # Reference: dequantize + matmul
    dequantized = awq_dequantize_triton(qweight, scales, qzeros)
    output_ref = torch.matmul(input_tensor.unsqueeze(0), dequantized).squeeze(0)

    # Calculate theoretical bytes moved
    bytes_moved = (
        K * 2  # input fp16
        + K * (N // 8) * 4  # qweight int32
        + num_groups * (N // 8) * 4  # qzeros int32
        + num_groups * N * 2  # scales fp16
        + N * 2  # output fp16
    )
    print(f"Theoretical data movement: {bytes_moved / 1e6:.2f} MB")

    # Test simple kernel with various configs - focus on best performers
    # (name, block_n, num_warps, unroll, num_stages)
    configs_simple = [
        # Best configs - u=32 without flatten is fastest
        ("bn=256, w=2, u=32, s=1", 256, 2, 32, 1),
        ("bn=512, w=4, u=32, s=1", 512, 4, 32, 1),
        ("bn=128, w=1, u=32, s=1", 128, 1, 32, 1),
        # Try u=16 to see if smaller unroll is even better
        ("bn=256, w=2, u=16, s=1", 256, 2, 16, 1),
        ("bn=512, w=4, u=16, s=1", 512, 4, 16, 1),
        ("bn=128, w=1, u=16, s=1", 128, 1, 16, 1),
        # Try u=8
        ("bn=256, w=2, u=8, s=1", 256, 2, 8, 1),
    ]

    results = []

    for name, block_n, num_warps, unroll, num_stages in configs_simple:
        try:
            if N % block_n != 0:
                continue
            if group_size % unroll != 0:
                continue

            output = awq_gemv_simple(
                input_tensor, qweight, qzeros, scales,
                block_n=block_n, num_warps=num_warps, unroll=unroll, num_stages=num_stages
            )

            diff = (output.squeeze() - output_ref).abs()
            max_diff = diff.max().item()
            correct = max_diff < 0.5

            if not correct:
                print(f"{name}: INCORRECT (max_diff={max_diff:.4f})")
                continue

            def run():
                return awq_gemv_simple(
                    input_tensor, qweight, qzeros, scales,
                    block_n=block_n, num_warps=num_warps, unroll=unroll, num_stages=num_stages
                )

            from vllm.triton_utils import triton
            ms = triton.testing.do_bench(run, warmup=25, rep=100, return_mode="min")
            bw = bytes_moved / (ms * 1e6)

            results.append((name, ms, bw, max_diff))
            print(f"{name}: {ms*1000:.1f} us, {bw:.1f} GB/s")

        except Exception as e:
            print(f"{name}: ERROR - {e}")

    # Test v2 split-K kernel
    print("\n--- Split-K kernel (v2) ---")
    v2_configs = [
        ("v2 sk=32, bn=256, w=4", 32, 256, 4),
        ("v2 sk=16, bn=256, w=4", 16, 256, 4),
        ("v2 sk=8, bn=256, w=4", 8, 256, 4),
    ]
    for name, split_k, block_n, num_warps in v2_configs:
        try:
            if num_groups % split_k != 0:
                continue
            if N % block_n != 0:
                continue

            output = awq_gemv_hip_style_v2(
                input_tensor, qweight, qzeros, scales,
                split_k=split_k, block_n=block_n, num_warps=num_warps
            )

            diff = (output.squeeze() - output_ref).abs()
            max_diff = diff.max().item()
            correct = max_diff < 0.5

            if not correct:
                print(f"{name}: INCORRECT (max_diff={max_diff:.4f})")
                continue

            def run():
                return awq_gemv_hip_style_v2(
                    input_tensor, qweight, qzeros, scales,
                    split_k=split_k, block_n=block_n, num_warps=num_warps
                )

            from vllm.triton_utils import triton
            ms = triton.testing.do_bench(run, warmup=25, rep=100, return_mode="min")
            bw = bytes_moved / (ms * 1e6)

            results.append((name, ms, bw, max_diff))
            print(f"{name}: {ms*1000:.1f} us, {bw:.1f} GB/s")

        except Exception as e:
            print(f"{name}: ERROR - {e}")

    # Test HIP kernel if available
    try:
        from vllm._custom_ops import awq_gemv_hip

        output_hip = awq_gemv_hip(input_tensor, qweight, scales, qzeros)
        diff = (output_hip - output_ref).abs()
        max_diff = diff.max().item()

        def run_hip():
            return awq_gemv_hip(input_tensor, qweight, scales, qzeros)

        from vllm.triton_utils import triton
        ms = triton.testing.do_bench(run_hip, warmup=25, rep=100, return_mode="min")
        bw = bytes_moved / (ms * 1e6)

        print(f"\nHIP kernel: {ms*1000:.1f} us, {bw:.1f} GB/s (max_diff={max_diff:.4f})")
        results.append(("HIP kernel", ms, bw, max_diff))
    except Exception as e:
        print(f"\nHIP kernel: not available ({e})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Kernel':<40} | {'Time':>10} | {'BW':>10} | {'Diff':>10}")
    print("-" * 80)
    for name, ms, bw, max_diff in sorted(results, key=lambda x: -x[2]):
        print(f"{name:<40} | {ms*1000:>8.1f} us | {bw:>8.1f} GB/s | {max_diff:>10.4f}")


def dump_triton_asm():
    """Dump Triton kernel assembly for analysis."""
    import os
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    N, K = 22016, 4096
    group_size = 128
    num_groups = K // group_size
    split_k = 8
    block_n = 256

    # Create test data
    torch.manual_seed(42)
    input_tensor = torch.randn(K, dtype=torch.float16, device="cuda")
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda")
    scales = torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01

    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device=input_tensor.device)
    num_n_blocks = N // block_n
    grid = (num_n_blocks, split_k)

    # Force compilation and get kernel
    kernel = awq_gemv_hip_style_v2_kernel[grid](
        input_tensor,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        SPLIT_K=split_k,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )

    # Get compiled kernel info
    print("\n" + "=" * 80)
    print("TRITON KERNEL ASSEMBLY")
    print("=" * 80)

    # Access compiled kernel asm
    try:
        compiled = awq_gemv_hip_style_v2_kernel.cache
        for key, val in compiled.items():
            if hasattr(val, 'asm'):
                print(f"Key: {key}")
                asm = val.asm
                if isinstance(asm, dict):
                    for asm_key, asm_val in asm.items():
                        if 'amdgcn' in asm_key or 'hsaco' in asm_key:
                            print(f"\n{asm_key}:")
                            print(asm_val[:5000] if len(asm_val) > 5000 else asm_val)
                else:
                    print(asm[:5000])
                break
    except Exception as e:
        print(f"Could not get ASM: {e}")

    # Alternative: use TRITON_CACHE_DIR
    print("\nTo get full assembly, run with:")
    print("TRITON_CACHE_DIR=/tmp/triton_asm python triton_awq_hip_style.py")
    print("Then look in /tmp/triton_asm for .amdgcn files")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--asm":
        dump_triton_asm()
    else:
        benchmark_kernels()
