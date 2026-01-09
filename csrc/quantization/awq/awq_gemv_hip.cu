// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// AWQ GEMV Kernel wrapper for ROCm/HIP
// The actual kernel is in benchmark_ddr/awq_gemv_kernel.hip

#ifdef USE_ROCM

  #include <torch/all.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPStream.h>
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>

  // IMPORTANT: Force assert to be active even with -DNDEBUG
  // The assert() call generates code that prevents the compiler from
  // over-optimizing and results in better register allocation and ~50% faster
  // code. See benchmark_ddr/vgpr_assert_minimal.hip for a minimal reproducer.
  #undef NDEBUG
  #include <cassert>

// ============================================================================
// AWQ GEMV Kernel - Quantized matrix-vector multiplication
// Software pipelined: 16 weight loads + 16 activation loads in flight
// Uses index-based buffer swapping for zeros/scales
// Accumulates in float16 (half precision)
//
// Template parameter OUTPUT_PER_THREAD controls load width:
//   32 = 128-bit loads (4 uint32 = 32 int4)
//   16 = 64-bit loads (2 uint32 = 16 int4)
//   8  = 32-bit loads (1 uint32 = 8 int4)
//
// Shapes (from AWQ):
//   activation - [K] (half)
//   qweight    - [K, N/8]  (8 int4 values packed per uint32)
//   scales     - [K/G, N] (half)
//   qzeros     - [K/G, N/8]
//   output     - [N] (half)
//
// Computation: output[n] = sum_k(activation[k] * (weight[k,n] - zero[k/G,n]) *
// scale[k/G,n])
// ============================================================================
template <int OUTPUT_PER_THREAD>
__global__ __launch_bounds__(256) void awq_gemv_kernel(
    const __half* __restrict__ activation,  // [K]
    const uint32_t* __restrict__ qweight,   // [K, N/8] - each uint32 has 8 int4
    const __half* __restrict__ scales,      // [K/G, N]
    const uint32_t* __restrict__ qzeros,    // [K/G, N/8]
    __half* __restrict__ output,            // [N]
    size_t M, size_t K, size_t N, size_t G) {
  static_assert(OUTPUT_PER_THREAD == 32 || OUTPUT_PER_THREAD == 16 ||
                    OUTPUT_PER_THREAD == 8,
                "OUTPUT_PER_THREAD must be 32, 16, or 8");

  // Derived constants
  constexpr int UINT32_PER_LOAD = OUTPUT_PER_THREAD / 8;  // 4, 2, or 1
  constexpr int LOAD_BYTES = UINT32_PER_LOAD * 4;         // 16, 8, or 4
  constexpr int PIPELINE_DEPTH = 16;
  constexpr int SCALES_UINT32_COUNT = OUTPUT_PER_THREAD / 2;

  // Number of groups and iterations per group (runtime)
  size_t NUM_GROUPS = K / G;
  size_t ITERS_PER_GROUP = G / PIPELINE_DEPTH;  // G=128: 8, G=64: 4, G=32: 2

  // This assert improves code generation by preventing over-optimization
  assert(NUM_GROUPS > 0 && "NUM_GROUPS must be positive");

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col_start = tid * OUTPUT_PER_THREAD;

  if (col_start >= N) return;

  // Use address_space(1) for global memory
  typedef const uint32_t __attribute__((address_space(1))) * global_uint32_ptr;

  // Accumulators in packed float16 (half2)
  // OUTPUT_PER_THREAD/2 half2 values = OUTPUT_PER_THREAD half values
  constexpr int ACC_HALF2_COUNT = OUTPUT_PER_THREAD / 2;
  __half2 acc2[ACC_HALF2_COUNT];
  #pragma unroll
  for (int i = 0; i < ACC_HALF2_COUNT; i++) {
    acc2[i] = __float2half2_rn(0.0f);
  }

  // ========== Pointer setup ==========
  // Weight pointer
  size_t weight_row_stride = (N / 8) * sizeof(uint32_t);
  size_t weight_ptr_val = reinterpret_cast<size_t>(qweight) + tid * LOAD_BYTES;
  uint32_t w_ptr_lo = static_cast<uint32_t>(weight_ptr_val);
  uint32_t w_ptr_hi = static_cast<uint32_t>(weight_ptr_val >> 32);
  uint32_t w_stride_lo = static_cast<uint32_t>(weight_row_stride);
  uint32_t w_stride_hi = static_cast<uint32_t>(weight_row_stride >> 32);

  // Zeros pointer
  size_t zeros_row_stride = (N / 8) * sizeof(uint32_t);
  size_t zeros_ptr_val = reinterpret_cast<size_t>(qzeros) +
                         tid * UINT32_PER_LOAD * sizeof(uint32_t);

  // Scales pointer
  size_t scales_row_stride = N * sizeof(__half);
  size_t scales_ptr_val =
      reinterpret_cast<size_t>(scales) + col_start * sizeof(__half);

  // Activation: use L2-cached global memory (8KB fits in L2)
  // Store activation pointer for direct access
  const __half* act_ptr = activation;

  #define W_PTR_ADD_ROW()                              \
    asm volatile(                                      \
        "v_add_co_u32 %0, vcc_lo, %0, %2\n\t"          \
        "v_add_co_ci_u32_e64 %1, null, %1, %3, vcc_lo" \
        : "+v"(w_ptr_lo), "+v"(w_ptr_hi)               \
        : "v"(w_stride_lo), "v"(w_stride_hi))

  #define GET_W_PTR()                    \
    reinterpret_cast<global_uint32_ptr>( \
        (static_cast<size_t>(w_ptr_hi) << 32) | w_ptr_lo)

  // ========== Pipeline registers ==========
  // Weight pipeline: 16 slots
  uint32_t w[PIPELINE_DEPTH][UINT32_PER_LOAD];

  // Double-buffered zeros and scales (use index to swap, not copy)
  // Buffer 0 and Buffer 1 - stored as half2 for packed operations
  uint32_t packed_zeros[2][UINT32_PER_LOAD];
  __half2 zeros2[2][ACC_HALF2_COUNT];   // Packed zeros
  __half2 scales2[2][ACC_HALF2_COUNT];  // Packed scales
  int curr_buf = 0;                     // Index of current buffer (0 or 1)

  // Row counter for weight/activation processing
  size_t weight_row = 0;

  // ========== Helper macros ==========
  #define LOAD_ZEROS_TO_BUF(group_idx, buf_idx)                        \
    do {                                                               \
      global_uint32_ptr zp = reinterpret_cast<global_uint32_ptr>(      \
          zeros_ptr_val + (group_idx) * zeros_row_stride);             \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {    \
        packed_zeros[buf_idx][j] = __builtin_nontemporal_load(zp + j); \
      }                                                                \
    } while (0)

  #define EXTRACT_ZEROS_IN_BUF(buf_idx)                                       \
    do {                                                                      \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {           \
        _Pragma("unroll") for (int b = 0; b < 4; b++) {                       \
          /* Extract 2 adjacent int4 values and pack into half2 */            \
          /* Use __ushort2half_rn for direct u16->f16 conversion              \
           * (v_cvt_f16_u16) */                                               \
          /* AWQ uses interleaved packing: shifts = (n_in_pack // 2) * 4 +    \
           * (n_in_pack % 2) * 16 */                                          \
          /* To get linear pairs (0,1), (2,3), (4,5), (6,7): use shifts (b*4, \
           * b*4+16) */                                                       \
          uint16_t zero0 = static_cast<uint16_t>(                             \
              (packed_zeros[buf_idx][j] >> (b * 4)) & 0xF);                   \
          uint16_t zero1 = static_cast<uint16_t>(                             \
              (packed_zeros[buf_idx][j] >> (b * 4 + 16)) & 0xF);              \
          zeros2[buf_idx][j * 4 + b] = __halves2half2(                        \
              __ushort2half_rn(zero0), __ushort2half_rn(zero1));              \
        }                                                                     \
      }                                                                       \
    } while (0)

  #define LOAD_SCALES_TO_BUF(group_idx, buf_idx)                          \
    do {                                                                  \
      global_uint32_ptr sp = reinterpret_cast<global_uint32_ptr>(         \
          scales_ptr_val + (group_idx) * scales_row_stride);              \
      _Pragma("unroll") for (int j = 0; j < SCALES_UINT32_COUNT; j++) {   \
        /* Each uint32 contains 2 half values - load as half2 directly */ \
        uint32_t packed = __builtin_nontemporal_load(sp + j);             \
        scales2[buf_idx][j] = *reinterpret_cast<const __half2*>(&packed); \
      }                                                                   \
    } while (0)

  // Preloaded activations for current iteration
  __half2 act_preload[PIPELINE_DEPTH];

  // Load all 16 activations for the next iteration into registers from
  // L2-cached global memory
  #define PRELOAD_ACTIVATIONS(act_row_base)                                 \
    do {                                                                    \
      _Pragma("unroll") for (int slot = 0; slot < PIPELINE_DEPTH; slot++) { \
        act_preload[slot] = __half2half2(act_ptr[(act_row_base) + slot]);   \
      }                                                                     \
    } while (0)

  // Accumulate using current buffer: output += activation * (weight - zero) *
  // scale Uses packed half2 operations for 2x throughput Uses preloaded
  // activations from act_preload[] Uses __ushort2half_rn for direct u16->f16
  // conversion (v_cvt_f16_u16)
  #define ACCUMULATE_SLOT(slot)                                               \
    do {                                                                      \
      __half2 act2 = act_preload[slot];                                       \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {           \
        _Pragma("unroll") for (int b = 0; b < 4; b++) {                       \
          /* Extract 2 adjacent int4 values */                                \
          /* AWQ uses interleaved packing: shifts = (n_in_pack // 2) * 4 +    \
           * (n_in_pack % 2) * 16 */                                          \
          /* To get linear pairs (0,1), (2,3), (4,5), (6,7): use shifts (b*4, \
           * b*4+16) */                                                       \
          uint16_t w0 = static_cast<uint16_t>((w[slot][j] >> (b * 4)) & 0xF); \
          uint16_t w1 =                                                       \
              static_cast<uint16_t>((w[slot][j] >> (b * 4 + 16)) & 0xF);      \
          __half2 weight2 =                                                   \
              __halves2half2(__ushort2half_rn(w0), __ushort2half_rn(w1));     \
          /* dequant = (weight - zero) * scale */                             \
          __half2 dequant2 =                                                  \
              __hmul2(__hsub2(weight2, zeros2[curr_buf][j * 4 + b]),          \
                      scales2[curr_buf][j * 4 + b]);                          \
          /* acc += activation * dequant */                                   \
          acc2[j * 4 + b] = __hfma2(act2, dequant2, acc2[j * 4 + b]);         \
        }                                                                     \
      }                                                                       \
    } while (0)

  // ========== PROLOGUE: Load group 0 zeros/scales and first 16
  // weight/activation rows ========== Load zeros for group 0 into buffer 0
  LOAD_ZEROS_TO_BUF(0, 0);
  EXTRACT_ZEROS_IN_BUF(0);

  // Load scales for group 0 into buffer 0
  LOAD_SCALES_TO_BUF(0, 0);

  // Load first 16 weight rows
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    global_uint32_ptr p = GET_W_PTR();
  #pragma unroll
    for (int j = 0; j < UINT32_PER_LOAD; j++) {
      w[slot][j] = __builtin_nontemporal_load(p + j);
    }
    W_PTR_ADD_ROW();
  }
  weight_row = PIPELINE_DEPTH;  // We've loaded rows 0..15

  // Prefetch zeros and scales for group 1 into buffer 1
  LOAD_ZEROS_TO_BUF(1, 1);
  LOAD_SCALES_TO_BUF(1, 1);

  // ========== MAIN LOOP: Process all groups ==========
  size_t act_row_base = 0;  // Row index for activations (matches weight rows)

  // Preload first batch of activations
  PRELOAD_ACTIVATIONS(act_row_base);

  for (size_t group = 0; group < NUM_GROUPS - 1; group++) {
    // Process ITERS_PER_GROUP iterations for current group
    for (size_t inner = 0; inner < ITERS_PER_GROUP; inner++) {
  // Accumulate using preloaded activations
  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        ACCUMULATE_SLOT(slot);
      }
      act_row_base += PIPELINE_DEPTH;

      // Preload activations for next iteration
      PRELOAD_ACTIVATIONS(act_row_base);

  // Load next 16 weights
  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        global_uint32_ptr p = GET_W_PTR();
  #pragma unroll
        for (int j = 0; j < UINT32_PER_LOAD; j++) {
          w[slot][j] = __builtin_nontemporal_load(p + j);
        }
        W_PTR_ADD_ROW();
      }
      weight_row += PIPELINE_DEPTH;
    }

    // Extract zeros for next buffer (loads were already issued)
    int next_buf = 1 - curr_buf;
    EXTRACT_ZEROS_IN_BUF(next_buf);

    // Swap buffers (just flip index, no data copy!)
    curr_buf = next_buf;

    // Prefetch zeros/scales for group+2 into the now-free buffer
    if (group + 2 < NUM_GROUPS) {
      int prefetch_buf = 1 - curr_buf;
      LOAD_ZEROS_TO_BUF(group + 2, prefetch_buf);
      LOAD_SCALES_TO_BUF(group + 2, prefetch_buf);
    }
  }

  // ========== Last group: ITERS_PER_GROUP - 1 iterations + epilogue ==========
  for (size_t inner = 0; inner < ITERS_PER_GROUP - 1; inner++) {
  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      ACCUMULATE_SLOT(slot);
    }
    act_row_base += PIPELINE_DEPTH;

    // Preload activations for next iteration
    PRELOAD_ACTIVATIONS(act_row_base);

  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      global_uint32_ptr p = GET_W_PTR();
  #pragma unroll
      for (int j = 0; j < UINT32_PER_LOAD; j++) {
        w[slot][j] = __builtin_nontemporal_load(p + j);
      }
      W_PTR_ADD_ROW();
    }
    weight_row += PIPELINE_DEPTH;
  }

  // ========== EPILOGUE: Drain final 16 weights ==========
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    ACCUMULATE_SLOT(slot);
  }

  #undef PRELOAD_ACTIVATIONS
  #undef LOAD_ZEROS_TO_BUF
  #undef EXTRACT_ZEROS_IN_BUF
  #undef LOAD_SCALES_TO_BUF
  #undef LOAD_ACT2
  #undef ACCUMULATE_SLOT
  #undef GET_W_PTR
  #undef W_PTR_ADD_ROW

  // ========== Write outputs (unpack half2 to individual halves) ==========
  #pragma unroll
  for (int i = 0; i < ACC_HALF2_COUNT; i++) {
    size_t col0 = col_start + i * 2;
    size_t col1 = col_start + i * 2 + 1;
    if (col0 < N) {
      output[col0] = __low2half(acc2[i]);
    }
    if (col1 < N) {
      output[col1] = __high2half(acc2[i]);
    }
  }
}

// ============================================================================
// Split-K version of AWQ GEMV Kernel
// SPLIT_K threads work on the same output columns, each processing K/SPLIT_K
// rows Final reduction in shared memory Supported: SPLIT_K = 2, 4, 8
// ============================================================================
template <int OUTPUT_PER_THREAD, int SPLIT_K>
__global__ __launch_bounds__(256) void awq_gemv_kernel_splitk(
    const __half* __restrict__ activation,  // [K]
    const uint32_t* __restrict__ qweight,   // [K, N/8]
    const __half* __restrict__ scales,      // [K/G, N]
    const uint32_t* __restrict__ qzeros,    // [K/G, N/8]
    __half* __restrict__ output,            // [N]
    size_t M, size_t K, size_t N, size_t G) {
  static_assert(OUTPUT_PER_THREAD == 8,
                "Split-K only supports OUTPUT_PER_THREAD=8");
  static_assert(SPLIT_K == 2 || SPLIT_K == 4 || SPLIT_K == 8,
                "SPLIT_K must be 2, 4, or 8");

  // Thread organization: SPLIT_K splits of 32 threads each
  // SPLIT_K=2: 64 threads/block, SPLIT_K=4: 128 threads/block, SPLIT_K=8: 256
  // threads/block
  constexpr int THREADS_PER_SPLIT = 32;
  constexpr int UINT32_PER_LOAD = OUTPUT_PER_THREAD / 8;  // 1
  constexpr int PIPELINE_DEPTH = 16;
  constexpr int ACC_HALF2_COUNT = OUTPUT_PER_THREAD / 2;  // 4

  // Runtime number of groups and iterations per group
  size_t TOTAL_GROUPS = K / G;
  size_t GROUPS_PER_SPLIT = TOTAL_GROUPS / SPLIT_K;
  size_t ITERS_PER_GROUP = G / PIPELINE_DEPTH;  // G=128: 8, G=64: 4, G=32: 2

  assert(TOTAL_GROUPS > 0 && "TOTAL_GROUPS must be positive");
  assert(TOTAL_GROUPS % SPLIT_K == 0 &&
         "TOTAL_GROUPS must be divisible by SPLIT_K");

  // Determine which split this thread belongs to
  int split_id = threadIdx.x / THREADS_PER_SPLIT;         // 0 or 1
  int thread_in_split = threadIdx.x % THREADS_PER_SPLIT;  // 0-31

  // Calculate column assignment (same for both splits in a pair)
  size_t tid = blockIdx.x * THREADS_PER_SPLIT + thread_in_split;
  size_t col_start = tid * OUTPUT_PER_THREAD;

  if (col_start >= N) return;

  // Shared memory for reduction (fp32 for precision)
  extern __shared__ float
      smem_f[];  // [SPLIT_K][THREADS_PER_SPLIT][OUTPUT_PER_THREAD]
  float* my_smem = &smem_f[split_id * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                           thread_in_split * OUTPUT_PER_THREAD];

  typedef const uint32_t __attribute__((address_space(1))) * global_uint32_ptr;

  // Accumulators in fp32 for precision
  float acc[OUTPUT_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
    acc[i] = 0.0f;
  }

  // Starting group for this split
  size_t start_group = split_id * GROUPS_PER_SPLIT;
  size_t start_row = start_group * G;  // G elements per group

  // Pointer setup - offset to starting row
  size_t weight_row_stride = (N / 8) * sizeof(uint32_t);
  size_t weight_ptr_val = reinterpret_cast<size_t>(qweight) +
                          tid * UINT32_PER_LOAD * sizeof(uint32_t) +
                          start_row * weight_row_stride;
  uint32_t w_ptr_lo = static_cast<uint32_t>(weight_ptr_val);
  uint32_t w_ptr_hi = static_cast<uint32_t>(weight_ptr_val >> 32);
  uint32_t w_stride_lo = static_cast<uint32_t>(weight_row_stride);
  uint32_t w_stride_hi = static_cast<uint32_t>(weight_row_stride >> 32);

  size_t zeros_row_stride = (N / 8) * sizeof(uint32_t);
  size_t zeros_ptr_val = reinterpret_cast<size_t>(qzeros) +
                         tid * UINT32_PER_LOAD * sizeof(uint32_t);

  size_t scales_row_stride = N * sizeof(__half);
  size_t scales_ptr_val =
      reinterpret_cast<size_t>(scales) + col_start * sizeof(__half);

  const __half* act_ptr = activation + start_row;

  #define W_PTR_ADD_ROW_SK()                           \
    asm volatile(                                      \
        "v_add_co_u32 %0, vcc_lo, %0, %2\n\t"          \
        "v_add_co_ci_u32_e64 %1, null, %1, %3, vcc_lo" \
        : "+v"(w_ptr_lo), "+v"(w_ptr_hi)               \
        : "v"(w_stride_lo), "v"(w_stride_hi))

  #define GET_W_PTR_SK()                 \
    reinterpret_cast<global_uint32_ptr>( \
        (static_cast<size_t>(w_ptr_hi) << 32) | w_ptr_lo)

  // Pipeline registers
  uint32_t w[PIPELINE_DEPTH][UINT32_PER_LOAD];
  uint32_t packed_zeros[2][UINT32_PER_LOAD];
  __half2 zeros2[2][ACC_HALF2_COUNT];
  __half2 scales2[2][ACC_HALF2_COUNT];
  int curr_buf = 0;

  // Preloaded activations
  __half2 act2[PIPELINE_DEPTH / 2];

  #define LOAD_ACT2_SK(idx)                           \
    __halves2half2(act_ptr[act_row_base + (idx) * 2], \
                   act_ptr[act_row_base + (idx) * 2 + 1])

  #define PRELOAD_ACTIVATIONS_SK(base)                                 \
    do {                                                               \
      _Pragma("unroll") for (int i = 0; i < PIPELINE_DEPTH / 2; i++) { \
        act2[i] = LOAD_ACT2_SK(i);                                     \
      }                                                                \
    } while (0)

  #define LOAD_ZEROS_TO_BUF_SK(group_idx, buf_idx)                     \
    do {                                                               \
      global_uint32_ptr zp = reinterpret_cast<global_uint32_ptr>(      \
          zeros_ptr_val + (group_idx) * zeros_row_stride);             \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {    \
        packed_zeros[buf_idx][j] = __builtin_nontemporal_load(zp + j); \
      }                                                                \
    } while (0)

  #define EXTRACT_ZEROS_IN_BUF_SK(buf_idx)                          \
    do {                                                            \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) { \
        _Pragma("unroll") for (int b = 0; b < 4; b++) {             \
          uint16_t zero0 = static_cast<uint16_t>(                   \
              (packed_zeros[buf_idx][j] >> (b * 4)) & 0xF);         \
          uint16_t zero1 = static_cast<uint16_t>(                   \
              (packed_zeros[buf_idx][j] >> (b * 4 + 16)) & 0xF);    \
          zeros2[buf_idx][j * 4 + b] = __halves2half2(              \
              __ushort2half_rn(zero0), __ushort2half_rn(zero1));    \
        }                                                           \
      }                                                             \
    } while (0)

  #define LOAD_SCALES_TO_BUF_SK(group_idx, buf_idx)                     \
    do {                                                                \
      const __half* sp = reinterpret_cast<const __half*>(               \
          scales_ptr_val + (group_idx) * scales_row_stride);            \
      _Pragma("unroll") for (int i = 0; i < ACC_HALF2_COUNT; i++) {     \
        scales2[buf_idx][i] = __halves2half2(sp[i * 2], sp[i * 2 + 1]); \
      }                                                                 \
    } while (0)

  #define ACCUMULATE_SLOT_SK(slot)                                            \
    do {                                                                      \
      __half2 a2 = act2[(slot) / 2];                                          \
      __half act_val = ((slot) % 2 == 0) ? __low2half(a2) : __high2half(a2);  \
      float act_f = __half2float(act_val);                                    \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {           \
        _Pragma("unroll") for (int b = 0; b < 4; b++) {                       \
          uint16_t w0 = static_cast<uint16_t>((w[slot][j] >> (b * 4)) & 0xF); \
          uint16_t w1 =                                                       \
              static_cast<uint16_t>((w[slot][j] >> (b * 4 + 16)) & 0xF);      \
          __half2 weight2 =                                                   \
              __halves2half2(__ushort2half_rn(w0), __ushort2half_rn(w1));     \
          /* dequant = (weight - zero) * scale in fp32 */                     \
          __half2 z2 = zeros2[curr_buf][j * 4 + b];                           \
          __half2 s2 = scales2[curr_buf][j * 4 + b];                          \
          float dequant0 = (__half2float(__low2half(weight2)) -               \
                            __half2float(__low2half(z2))) *                   \
                           __half2float(__low2half(s2));                      \
          float dequant1 = (__half2float(__high2half(weight2)) -              \
                            __half2float(__high2half(z2))) *                  \
                           __half2float(__high2half(s2));                     \
          /* acc += activation * dequant in fp32 */                           \
          acc[(j * 4 + b) * 2] += act_f * dequant0;                           \
          acc[(j * 4 + b) * 2 + 1] += act_f * dequant1;                       \
        }                                                                     \
      }                                                                       \
    } while (0)

  // Load zeros and scales for first group of this split
  LOAD_ZEROS_TO_BUF_SK(start_group, 0);
  EXTRACT_ZEROS_IN_BUF_SK(0);
  LOAD_SCALES_TO_BUF_SK(start_group, 0);

  // Load first 16 weight rows
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
    for (int j = 0; j < UINT32_PER_LOAD; j++) {
      w[slot][j] = __builtin_nontemporal_load(p + j);
    }
    W_PTR_ADD_ROW_SK();
  }

  // Prefetch zeros/scales for next group
  if (start_group + 1 < start_group + GROUPS_PER_SPLIT) {
    LOAD_ZEROS_TO_BUF_SK(start_group + 1, 1);
    LOAD_SCALES_TO_BUF_SK(start_group + 1, 1);
  }

  size_t act_row_base = 0;
  PRELOAD_ACTIVATIONS_SK(act_row_base);

  // Main loop: process GROUPS_PER_SPLIT groups
  for (size_t group = 0; group < GROUPS_PER_SPLIT - 1; group++) {
    for (size_t inner = 0; inner < ITERS_PER_GROUP; inner++) {
  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        ACCUMULATE_SLOT_SK(slot);
      }
      act_row_base += PIPELINE_DEPTH;
      PRELOAD_ACTIVATIONS_SK(act_row_base);

  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
        for (int j = 0; j < UINT32_PER_LOAD; j++) {
          w[slot][j] = __builtin_nontemporal_load(p + j);
        }
        W_PTR_ADD_ROW_SK();
      }
    }

    int next_buf = 1 - curr_buf;
    EXTRACT_ZEROS_IN_BUF_SK(next_buf);
    curr_buf = next_buf;

    if (group + 2 < GROUPS_PER_SPLIT) {
      int prefetch_buf = 1 - curr_buf;
      LOAD_ZEROS_TO_BUF_SK(start_group + group + 2, prefetch_buf);
      LOAD_SCALES_TO_BUF_SK(start_group + group + 2, prefetch_buf);
    }
  }

  // Last group
  for (size_t inner = 0; inner < ITERS_PER_GROUP - 1; inner++) {
  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      ACCUMULATE_SLOT_SK(slot);
    }
    act_row_base += PIPELINE_DEPTH;
    PRELOAD_ACTIVATIONS_SK(act_row_base);

  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
      for (int j = 0; j < UINT32_PER_LOAD; j++) {
        w[slot][j] = __builtin_nontemporal_load(p + j);
      }
      W_PTR_ADD_ROW_SK();
    }
  }

  // Epilogue
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    ACCUMULATE_SLOT_SK(slot);
  }

  #undef PRELOAD_ACTIVATIONS_SK
  #undef LOAD_ZEROS_TO_BUF_SK
  #undef EXTRACT_ZEROS_IN_BUF_SK
  #undef LOAD_SCALES_TO_BUF_SK
  #undef ACCUMULATE_SLOT_SK
  #undef LOAD_ACT2_SK
  #undef GET_W_PTR_SK
  #undef W_PTR_ADD_ROW_SK

  // Store partial results to shared memory (fp32)
  #pragma unroll
  for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
    my_smem[i] = acc[i];
  }

  __syncthreads();

  // Tree reduction across splits (in fp32)
  // For SPLIT_K=2: one reduction step
  // For SPLIT_K=4: two reduction steps
  // For SPLIT_K=8: three reduction steps
  if constexpr (SPLIT_K >= 8) {
    // Step 1: splits 0-3 add splits 4-7
    if (split_id < 4) {
      float* other_smem =
          &smem_f[(split_id + 4) * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                  thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
      for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
        acc[i] = my_smem[i] + other_smem[i];
        my_smem[i] = acc[i];
      }
    }
    __syncthreads();
  }

  if constexpr (SPLIT_K >= 4) {
    // Step 2: splits 0-1 add splits 2-3
    if (split_id < 2) {
      float* other_smem =
          &smem_f[(split_id + 2) * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                  thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
      for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
        acc[i] = my_smem[i] + other_smem[i];
        my_smem[i] = acc[i];
      }
    }
    __syncthreads();
  }

  // Final step: split 0 adds split 1
  if (split_id == 0) {
    float* other_smem = &smem_f[1 * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                                thread_in_split * OUTPUT_PER_THREAD];

  #pragma unroll
    for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
      acc[i] = my_smem[i] + other_smem[i];
    }

    // Write outputs (convert fp32 accumulators to fp16)
  #pragma unroll
    for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
      size_t col = col_start + i;
      if (col < N) {
        output[col] = __float2half(acc[i]);
      }
    }
  }
}

// PyTorch binding wrapper
torch::Tensor awq_gemv_hip(torch::Tensor activation,  // [M, K] or [K]
                           torch::Tensor qweight,     // [K, N/8]
                           torch::Tensor scales,      // [K/G, N]
                           torch::Tensor qzeros)      // [K/G, N/8]
{
  // ========== Dimension checks ==========
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D, got ", qweight.dim(),
              "D");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be 2D, got ", qzeros.dim(), "D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D, got ", scales.dim(), "D");

  // Get dimensions from qweight (the authoritative source)
  int64_t K = qweight.size(0);
  int64_t N = qweight.size(1) * 8;  // Each uint32 packs 8 int4 values
  int64_t num_groups = qzeros.size(0);

  TORCH_CHECK(num_groups > 0, "num_groups must be positive, got ", num_groups);
  TORCH_CHECK(K % num_groups == 0, "K (", K,
              ") must be divisible by num_groups (", num_groups, ")");

  int64_t G = K / num_groups;  // Group size

  // ========== M=1 (GEMV) constraint ==========
  TORCH_CHECK(activation.dim() == 1 ||
                  (activation.dim() == 2 && activation.size(0) == 1),
              "awq_gemv_hip only supports M=1 (GEMV), got activation shape ",
              activation.sizes());

  // ========== Group size constraint ==========
  // Kernel uses PIPELINE_DEPTH=16, so G must be divisible by 16
  // Currently we only support G=128 for optimal performance
  TORCH_CHECK(G == 128, "awq_gemv_hip only supports group_size=128, got ", G);
  TORCH_CHECK(G % 16 == 0, "group_size (", G,
              ") must be divisible by PIPELINE_DEPTH (16)");

  // ========== N dimension constraints ==========
  // N must be divisible by 8 for weight packing (8 int4 per uint32)
  TORCH_CHECK(N % 8 == 0, "N (", N, ") must be divisible by 8");

  // ========== Activation size check ==========
  int64_t act_size =
      activation.dim() == 1 ? activation.size(0) : activation.size(1);
  TORCH_CHECK(act_size >= K, "activation size (", act_size, ") must be >= K (",
              K, ")");

  // ========== Scales shape validation ==========
  // scales must be [K/G, N] = [num_groups, N]
  TORCH_CHECK(scales.size(0) == num_groups, "scales.size(0) (", scales.size(0),
              ") must equal num_groups (", num_groups, ")");
  TORCH_CHECK(scales.size(1) == N, "scales.size(1) (", scales.size(1),
              ") must equal N (", N, ")");

  // ========== Qzeros shape validation ==========
  // qzeros must be [K/G, N/8] = [num_groups, N/8]
  TORCH_CHECK(qzeros.size(0) == num_groups, "qzeros.size(0) (", qzeros.size(0),
              ") must equal num_groups (", num_groups, ")");
  TORCH_CHECK(qzeros.size(1) == N / 8, "qzeros.size(1) (", qzeros.size(1),
              ") must equal N/8 (", N / 8, ")");

  // ========== Contiguity checks ==========
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(qzeros.is_contiguous(), "qzeros must be contiguous");

  // ========== Dtype checks ==========
  TORCH_CHECK(activation.scalar_type() == at::ScalarType::Half,
              "activation must be float16, got ", activation.scalar_type());
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Half,
              "scales must be float16, got ", scales.scalar_type());
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Int,
              "qweight must be int32, got ", qweight.scalar_type());
  TORCH_CHECK(qzeros.scalar_type() == at::ScalarType::Int,
              "qzeros must be int32, got ", qzeros.scalar_type());

  // ========== Device checks ==========
  TORCH_CHECK(activation.is_cuda(), "activation must be on CUDA device");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be on CUDA device");
  TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA device");
  TORCH_CHECK(qzeros.is_cuda(), "qzeros must be on CUDA device");
  TORCH_CHECK(activation.device() == qweight.device(),
              "activation and qweight must be on the same device");
  TORCH_CHECK(scales.device() == qweight.device(),
              "scales and qweight must be on the same device");
  TORCH_CHECK(qzeros.device() == qweight.device(),
              "qzeros and qweight must be on the same device");

  // Create output tensor
  auto output = torch::empty({N}, scales.options());

  // Flatten activation if 2D
  auto act_flat = activation.dim() == 2 ? activation.squeeze(0) : activation;

  // Launch parameters
  constexpr int OUTPUT_PER_THREAD = 8;

  // Get current stream
  auto stream = at::hip::getCurrentHIPStream();

  // Choose kernel based on N and num_groups divisibility
  int64_t total_outputs = (N + OUTPUT_PER_THREAD - 1) / OUTPUT_PER_THREAD;
  constexpr int THREADS_PER_SPLIT = 32;

  // Check natural divisibility for split-k
  bool can_use_splitk_8 = (num_groups % 8 == 0);
  bool can_use_splitk_4 = (num_groups % 4 == 0);
  bool can_use_splitk_2 = (num_groups % 2 == 0);

  if (can_use_splitk_8 && N <= 16384) {
    // Use split-k=8 for maximum parallelism
    constexpr int SPLIT_K = 8;
    constexpr int THREADS_PER_BLOCK = THREADS_PER_SPLIT * SPLIT_K;  // 256
    int64_t num_blocks =
        (total_outputs + THREADS_PER_SPLIT - 1) / THREADS_PER_SPLIT;
    size_t smem_size =
        SPLIT_K * THREADS_PER_SPLIT * OUTPUT_PER_THREAD * sizeof(float);

    awq_gemv_kernel_splitk<8, SPLIT_K>
        <<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
            reinterpret_cast<const __half*>(act_flat.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), 1,
            static_cast<size_t>(K), static_cast<size_t>(N),
            static_cast<size_t>(G));
  } else if (can_use_splitk_4 && N <= 24576) {
    // Small N: use split-k=4 for more parallelism
    constexpr int SPLIT_K = 4;
    constexpr int THREADS_PER_BLOCK = THREADS_PER_SPLIT * SPLIT_K;  // 128
    int64_t num_blocks =
        (total_outputs + THREADS_PER_SPLIT - 1) / THREADS_PER_SPLIT;
    size_t smem_size =
        SPLIT_K * THREADS_PER_SPLIT * OUTPUT_PER_THREAD * sizeof(float);

    awq_gemv_kernel_splitk<8, SPLIT_K>
        <<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
            reinterpret_cast<const __half*>(act_flat.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), 1,
            static_cast<size_t>(K), static_cast<size_t>(N),
            static_cast<size_t>(G));
  } else if (can_use_splitk_2 && N <= 32768) {
    // Medium N: use split-k=2
    constexpr int SPLIT_K = 2;
    constexpr int THREADS_PER_BLOCK = THREADS_PER_SPLIT * SPLIT_K;  // 64
    int64_t num_blocks =
        (total_outputs + THREADS_PER_SPLIT - 1) / THREADS_PER_SPLIT;
    size_t smem_size =
        SPLIT_K * THREADS_PER_SPLIT * OUTPUT_PER_THREAD * sizeof(float);

    awq_gemv_kernel_splitk<8, SPLIT_K>
        <<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
            reinterpret_cast<const __half*>(act_flat.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), 1,
            static_cast<size_t>(K), static_cast<size_t>(N),
            static_cast<size_t>(G));
  } else {
    // No split-k: use regular kernel
    constexpr int THREADS_PER_BLOCK = 32;
    int64_t num_blocks =
        (total_outputs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    awq_gemv_kernel<8><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(act_flat.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        1,  // M
        static_cast<size_t>(K), static_cast<size_t>(N), static_cast<size_t>(G));
  }

  return output;
}

#endif  // USE_ROCM
