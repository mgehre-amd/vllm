# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements

SKINNY_GEMM_MAX_N = 4


def _w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_dequant: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM kernel and dequant fallback.

    Registered as a custom op so torch.compile treats it as opaque,
    avoiding issues with the data-dependent N<=4 branch.
    """
    import vllm._custom_ops as ops

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if N <= SKINNY_GEMM_MAX_N and K * N <= LDS_CAPACITY_ELEMENTS:
        if group_size > 0:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, bias)
        else:
            return ops.wvSplitK_int4(w_q, x_2d, w_s, cu_count, bias)

    assert w_dequant is not None, (
        "w_dequant must be pre-computed for large-batch fallback"
    )
    return torch.nn.functional.linear(x_2d, w_dequant, bias)


def _w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_dequant: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    N = x_2d.size(0)
    M = w_q.size(0)
    return torch.empty((N, M), dtype=x_2d.dtype, device=x_2d.device)


def _register_w4a16_op():
    lib = torch.library.Library("_rocm_skinny", "DEF")
    lib.define(
        "w4a16_apply(Tensor x_2d, Tensor w_q, Tensor w_s, Tensor? w_dequant,"
        " Tensor? bias, int cu_count, int group_size) -> Tensor"
    )
    lib.impl("w4a16_apply", _w4a16_apply_impl, "CUDA")
    lib.impl("w4a16_apply", _w4a16_apply_fake, "Meta")
    return lib


_W4A16_LIB = _register_w4a16_op()


class HipW4A16SkinnyLinearKernel(MPLinearKernel):
    """W4A16 skinny GEMM for ROCm (gfx11) using packed int4 weights.

    Supports both per-channel (group_size=-1) and per-group (group_size=32/128)
    symmetric quantization. Uses wvSplitK_int4/wvSplitK_int4_g for small batch
    sizes where activations fit in LDS, with dequant+linear fallback for larger
    batches. Wrapped as a custom op to avoid torch.compile issues with the
    data-dependent N<=4 branch.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int4"
            ):
                return False, "wvSplitK_int4 op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type.size_bits != 4:
            return False, "requires 4-bit weights"

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.group_size not in (-1, 32, 128):
            return False, f"group_size must be -1, 32, or 128 (got {c.group_size})"

        if c.zero_points:
            return False, "does not support zero points (asymmetric)"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if c.group_size > 0 and K % c.group_size != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={c.group_size}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        K = c.partition_weight_shape[0]

        # Dequantize from the raw unpacked representation *before* repacking
        # into the kernel-specific layout. This is simpler than reversing the
        # ExLlama shuffle / packed-byte format after transform_param.
        w_q_raw, w_s_raw, _, _ = self._get_weight_params(layer)
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        bias_val = c.weight_type.bias
        w_fp = (unpacked - bias_val).to(c.act_type)

        w_s_clean = w_s_raw.data
        if c.group_size == -1:
            w_s_clean = w_s_clean.squeeze(-1)
        w_s_clean = w_s_clean.contiguous()

        if c.group_size > 0:
            num_groups = K // c.group_size
            w_fp = w_fp.view(w_fp.shape[0], num_groups, c.group_size)
            w_fp = (w_fp * w_s_clean.unsqueeze(-1)).view(w_fp.shape[0], K)
        else:
            w_fp = w_fp * w_s_clean.unsqueeze(1)
        self._w_dequant = w_fp.contiguous()

        # Now repack w_q / w_s into the format the skinny GEMM kernel expects.
        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            unpacked = unpack_quantized_values_into_int32(
                x.data, c.weight_type, packed_dim=x.packed_dim
            )
            if c.act_type == torch.float16:
                unsigned = unpacked.to(torch.uint8)
                M, K_dim = unsigned.shape
                g = unsigned.view(M, K_dim // 8, 8).to(torch.int32)
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
                bias_val = c.weight_type.bias
                signed = (unpacked - bias_val).to(torch.int8)
                M, K_dim = signed.shape
                low = signed[:, 0::2] & 0xF
                high = signed[:, 1::2] & 0xF
                packed = (low | (high << 4)).to(torch.uint8)
                return packed.view(torch.int8).contiguous()

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            if c.group_size == -1:
                return x.data.squeeze(-1).contiguous()
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        w_q, w_s, _, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        N = x_2d.shape[0]
        K = x_2d.shape[1]
        M = w_q.shape[0]
        out_shape = x.shape[:-1] + (M,)

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"hip_w4a16_skinny {N}x{M}x{K}")
        )
        with ctx:
            cu_count = num_compute_units()
            output = torch.ops._rocm_skinny.w4a16_apply(
                x_2d,
                w_q,
                w_s,
                self._w_dequant,
                bias,
                cu_count,
                self.config.group_size,
            )
        return output.reshape(out_shape)
