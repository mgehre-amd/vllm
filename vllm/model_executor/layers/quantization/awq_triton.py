# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Set up the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def awq_gemv_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [N] fp16
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized N-split AWQ kernel WITHOUT masking.

    Assumes BLOCK_N divides N evenly to eliminate exec mask manipulation.
    This should generate much cleaner assembly.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    N_packed = N // 8
    num_groups = K // GROUP_SIZE

    # Output column indices [BLOCK_N] - NO MASK needed
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Packed column indices
    n_packed_offs = n_offs // 8
    n_in_pack = n_offs % 8

    # AWQ shift amounts
    shifts = (n_in_pack // 2) * 4 + (n_in_pack % 2) * 16

    # Accumulator tensor [BLOCK_N]
    acc = tl.zeros([BLOCK_N], dtype=tl.float16)

    for g in tl.range(num_groups, flatten=True):
        # Load scales [BLOCK_N] - NO MASK
        scale_ptrs = scales_ptr + g * N + n_offs
        scales = tl.load(scale_ptrs).to(tl.float16)

        # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
        qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
        qz = tl.load(qz_ptrs).to(tl.uint32)
        zeros = ((qz >> shifts) & 0xF).to(tl.float16)

        # Precompute bias
        bias = -zeros * scales

        # Inner loop over K in this group
        for k in tl.range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
            # Load input scalar and broadcast
            x = tl.load(input_ptr + k).to(tl.float16)

            # Load weights - NO MASK
            qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
            qw = tl.load(qw_ptrs).to(tl.uint32)
            w = ((qw >> shifts) & 0xF).to(tl.float16)

            # Accumulate
            acc += x * (w * scales + bias)

    # Store results - NO MASK
    tl.store(output_ptr + n_offs, acc.to(tl.float16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "SPLIT_K": 1}, num_warps=2),
        triton.Config({"BLOCK_N": 64, "SPLIT_K": 2}, num_warps=2),
        triton.Config({"BLOCK_N": 64, "SPLIT_K": 4}, num_warps=2),
        triton.Config({"BLOCK_N": 64, "SPLIT_K": 8}, num_warps=2),
        triton.Config({"BLOCK_N": 64, "SPLIT_K": 16}, num_warps=2),
        triton.Config({"BLOCK_N": 128, "SPLIT_K": 1}, num_warps=2),
        triton.Config({"BLOCK_N": 128, "SPLIT_K": 2}, num_warps=2),
        triton.Config({"BLOCK_N": 128, "SPLIT_K": 4}, num_warps=2),
        triton.Config({"BLOCK_N": 128, "SPLIT_K": 8}, num_warps=2),
        triton.Config({"BLOCK_N": 128, "SPLIT_K": 16}, num_warps=2),
        triton.Config({"BLOCK_N": 32, "SPLIT_K": 1}, num_warps=2),
        triton.Config({"BLOCK_N": 32, "SPLIT_K": 2}, num_warps=2),
        triton.Config({"BLOCK_N": 32, "SPLIT_K": 4}, num_warps=2),
        triton.Config({"BLOCK_N": 32, "SPLIT_K": 8}, num_warps=2),
        triton.Config({"BLOCK_N": 512, "SPLIT_K": 32}, num_warps=1),
    ],
    key=["K", "N"],
)
@triton.jit
def awq_gemv_kernel_split_k(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [N] fp16 or [split_k, N] fp16 if split_k > 1
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """
    Optimized N-split AWQ kernel WITHOUT masking, with optional split-K.

    Assumes BLOCK_N divides N evenly to eliminate exec mask manipulation.

    Split-K parallelization:
    - SPLIT_K=1: Normal operation, each workgroup processes all K
    - SPLIT_K>1: Each workgroup processes K/SPLIT_K elements, writes partial results
    """
    pid_n = tl.program_id(0)  # N dimension
    pid_k = tl.program_id(1)  # K split dimension (0 if SPLIT_K=1)

    n_start = pid_n * BLOCK_N

    N_packed = N // 8
    num_groups = K // GROUP_SIZE

    # Output column indices [BLOCK_N] - NO MASK needed
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Packed column indices
    n_packed_offs = n_offs // 8
    n_in_pack = n_offs % 8

    # AWQ shift amounts
    shifts = (n_in_pack // 2) * 4 + (n_in_pack % 2) * 16

    # Accumulator tensor [BLOCK_N]
    acc = tl.zeros([BLOCK_N], dtype=tl.float16)

    if SPLIT_K == 1:
        # Fast path: no split-K, use static loops
        for g in tl.range(num_groups, flatten=True):
            # Load scales [BLOCK_N] - NO MASK
            scale_ptrs = scales_ptr + g * N + n_offs
            scales = tl.load(scale_ptrs).to(tl.float16)

            # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
            qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
            qz = tl.load(qz_ptrs).to(tl.uint32)
            zeros = ((qz >> shifts) & 0xF).to(tl.float16)

            # Precompute bias
            bias = -zeros * scales

            # Inner loop over K in this group
            for k in tl.range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
                # Load input scalar and broadcast
                x = tl.load(input_ptr + k).to(tl.float16)

                # Load weights - NO MASK
                qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
                qw = tl.load(qw_ptrs).to(tl.uint32)
                w = ((qw >> shifts) & 0xF).to(tl.float16)

                # Accumulate
                acc += x * (w * scales + bias)
    else:
        # Split-K path: process subset of K
        k_per_split = K // SPLIT_K
        k_start = pid_k * k_per_split
        k_end = k_start + k_per_split

        # Group range for this split
        g_start = k_start // GROUP_SIZE
        g_end = (k_end + GROUP_SIZE - 1) // GROUP_SIZE  # Round up

        for g in tl.range(g_start, g_end):
            # Load scales [BLOCK_N] - NO MASK
            scale_ptrs = scales_ptr + g * N + n_offs
            scales = tl.load(scale_ptrs).to(tl.float16)

            # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
            qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
            qz = tl.load(qz_ptrs).to(tl.uint32)
            zeros = ((qz >> shifts) & 0xF).to(tl.float16)

            # Precompute bias
            bias = -zeros * scales

            # K range for this group, clipped to our split's K range
            k_group_start = tl.maximum(g * GROUP_SIZE, k_start)
            k_group_end = tl.minimum((g + 1) * GROUP_SIZE, k_end)

            # Inner loop over K in this group (within our split)
            for k in range(k_group_start, k_group_end):
                # Load input scalar and broadcast
                x = tl.load(input_ptr + k).to(tl.float16)

                # Load weights - NO MASK
                qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
                qw = tl.load(qw_ptrs).to(tl.uint32)
                w = ((qw >> shifts) & 0xF).to(tl.float16)

                # Accumulate
                acc += x * (w * scales + bias)

    # Store results - NO MASK
    if SPLIT_K == 1:
        # Direct write to output
        tl.store(output_ptr + n_offs, acc.to(tl.float16))
    else:
        # Write partial results: output[pid_k, n_offs]
        out_offs = pid_k * N + n_offs
        tl.store(output_ptr + out_offs, acc.to(tl.float16))


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def _awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # if M == 1 and N % 512 == 0 and N == 19456:
    # assert isinstance(M, int)

    if M == 1 and N % 512 == 0:
        result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)
        if N == 19456:
            BLOCK_N = 128
            gridA = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
            # with profiler.record_function(f"awq_gemv {N}x{K}"):
            if True:
                awq_gemv_kernel[gridA](
                    input,
                    qweight,
                    qzeros,
                    scales,
                    result,
                    K=K,
                    N=N,
                    GROUP_SIZE=group_size,
                    BLOCK_N=BLOCK_N,
                    num_warps=2,
                )
            return result
        else:
            BLOCK_N = 64
            gridA = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
            # with profiler.record_function(f"awq_gemv_kernel_split_k {N}x{K}"):
            if True:
                awq_gemv_kernel_split_k[gridA](
                    input,
                    qweight,
                    qzeros,
                    scales,
                    result,
                    K=K,
                    N=N,
                    GROUP_SIZE=group_size,
                )
            return result

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    # with profiler.record_function(f"awq_gemm {M}x{N}x{K}"):
    if True:
        awq_gemm_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            M,
            N,
            K,
            group_size,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            SPLIT_K=split_k_iters,
        )

        result = result.sum(0)

    return result


def _awq_gemm_triton_fake(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    M, N = input.shape[0], qweight.shape[1] * 8
    return torch.empty((M, N), dtype=scales.dtype, device=input.device)


direct_register_custom_op(
    op_name="awq_gemm_triton",
    op_func=_awq_gemm_triton,
    fake_impl=_awq_gemm_triton_fake,
)
awq_gemm_triton = torch.ops.vllm.awq_gemm_triton
