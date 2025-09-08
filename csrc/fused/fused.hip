/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../dispatch_utils.h"
#include "../utils.cuh"
#include "../reduction_utils.h"
// #include "../numeric_conversion.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

enum class QuantType
{
  kInt8,
  kInt4,
};

__device__ __forceinline__ float u32_as_f32(uint32_t u) {
  union { uint32_t u; float f; } v{u}; return v.f;
}
__device__ __forceinline__ uint32_t f32_as_u32(float f) {
  union { uint32_t u; float f; } v{0}; v.f = f; return v.u;
}
__device__ __forceinline__ uint16_t bf16_bits(__hip_bfloat16 x) {
  return *reinterpret_cast<const uint16_t*>(&x);
}
__device__ __forceinline__ __hip_bfloat16 bits_to_bf16(uint16_t b) {
  __hip_bfloat16 x; *reinterpret_cast<uint16_t*>(&x) = b; return x;
}


// ========== to-float ==========
template <typename T>
__device__ __forceinline__ float convert_to_float(T val);

// __half → float
template <>
__device__ __forceinline__ float convert_to_float<__half>(__half v) {
  return __half2float(v);
}

// __hip_bfloat16 → float（位级拼接，高 16 位）
template <>
__device__ __forceinline__ float convert_to_float<__hip_bfloat16>(__hip_bfloat16 v) {
  uint16_t hi = bf16_bits(v);
  return u32_as_f32(uint32_t(hi) << 16);
}

// hip_bfloat16 → float（通过重解释为 __hip_bfloat16）
template <>
__device__ __forceinline__ float convert_to_float<hip_bfloat16>(hip_bfloat16 v) {
  return convert_to_float(*reinterpret_cast<const __hip_bfloat16*>(&v));
}

// template <typename T>
// __device__ __forceinline__ float convert_to_float(T val) {
//   // static_assert(std::is_same<T, __half>::value ||
//   //               std::is_same<T, __hip_bfloat16>::value,
//   //               "Only __half and __hip_bfloat16 are supported (ROCm).");

//   if constexpr (std::is_same<T, __half>::value) {
//     // fp16 -> f32
//     return __half2float(val);
//   } else {
//     // bf16 -> f32
//     return __bfloat162float(val);
//   }
// }

template <typename T>
__device__ __forceinline__ T convert_from_float(float val) {
  static_assert(std::is_same<T, __half>::value ||
                std::is_same<T, __hip_bfloat16>::value,
                "Only __half and __hip_bfloat16 are supported (ROCm).");

  if constexpr (std::is_same<T, __half>::value) {
    // f32 -> fp16 (round-to-nearest)
    return __float2half_rn(val);
  } else {
    // f32 -> bf16 (round-to-nearest)
    return __float2bfloat16(val);
  }
}

namespace detail {

  struct vec16_t { float x, y, z, w; };

  template <bool PadZero, typename T>
  __device__ __forceinline__ void predicated_g2s_16B(T* smem_dst, const T* gmem_src, bool pred) {
    if (pred) {
      *reinterpret_cast<vec16_t*>(smem_dst) = *reinterpret_cast<const vec16_t*>(gmem_src);
    } else if constexpr (PadZero) {
      *reinterpret_cast<vec16_t*>(smem_dst) = vec16_t{0.f, 0.f, 0.f, 0.f};
    }
  }

  template<typename T>
  __device__ __forceinline__ void load_8xT_to_regs(const T* __restrict__ ptr, T (&dst)[8]) {
    static_assert(sizeof(T) == 2, "T must be 16-bit (half/bfloat16)");
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(ptr);
    *reinterpret_cast<uint4*>(&dst[0]) = *src;
  }

  __device__ __forceinline__ void store_8fp8(const uint32_t* __restrict__ fp8x4,
                                           int8_t* __restrict__ out) {
    *reinterpret_cast<uint2*>(out) = *reinterpret_cast<const uint2*>(fp8x4);
  }

  // ---- E4M3 打包：float32 -> uint8（rn、satfinite、非 subnormal）----
  // E4M3：1|4|3，bias=7；rn-even；satfinite；不产生 subnormal（下溢置0）
  __device__ __forceinline__ uint8_t float_to_e4m3_rn_satfinite_relaxed(float x) {
    // NaN/Inf -> satfinite（带符号），用你定义的“最大码”
    const uint8_t POS_MAX_CODE = 0x6F;  // sign=0, exp=14, mant=7  -> +240
    const uint8_t NEG_MAX_CODE = 0xEF;  // sign=1, exp=14, mant=7  -> -240

    if (!isfinite(x)) return signbit(x) ? NEG_MAX_CODE : POS_MAX_CODE;
    if (x == 0.0f)    return 0u;

    const uint8_t s = signbit(x) ? 0x80 : 0x00;
    float ax = fabsf(x);

    // 最小正规值 = 2^(1-bias) = 2^-6
    if (ax < (1.0f / 64.0f)) return s | 0x00;  // 不做 subnormal：直接 0

    // 规格化 ax = m * 2^e，m∈[1,2)
    int   e;
    float m = frexpf(ax, &e);   // m∈[0.5,1)
    m *= 2.0f; e -= 1;          // m∈[1,2)

    // 量化 3 位尾数（去掉隐含 1）：mant = rn_even((m-1)*8)
    float mant_f  = (m - 1.0f) * 8.0f;  // ∈[0,8)
    float floor_v = floorf(mant_f);
    float frac_v  = mant_f - floor_v;

    int mant;
    if      (frac_v > 0.5f) mant = (int)floor_v + 1;
    else if (frac_v < 0.5f) mant = (int)floor_v;
    else                    mant = ((int)floor_v & 1) ? (int)floor_v + 1 : (int)floor_v;

    // 尾数进位：mant==8 -> mant=0, e+1
    if (mant == 8) { mant = 0; e += 1; }

    // 带偏置指数
    int eb = e + 7;

    // 下溢：不做 subnormal，直接 0
    if (eb <= 0) return s | 0x00;

    // ------- 关键改动：指数=15 的处理规则 -------
    // 仅当 eb==15 且 mant==7 时，判定为“上溢”（你定义的溢出码）；
    // 其他 eb==15 且 mant!=7 的情况，按“有效数”编码（非常规做法，需上下游一致解码）。
    if (eb >= 0xF) {
      if (mant >= 7) {
        // eb==15 且 mant==7 -> 上溢（饱和到“最大码”）
        // 这里沿用你原有的“最大有限值”码（exp=14,mant=7），或你也可以选择返回 (15,7) 本码
        return s ? NEG_MAX_CODE : POS_MAX_CODE;
      } else {
        // eb==15 且 mant!=7 -> 视为有效数（非常规编码）
        uint8_t e_bits = 0xF;
        uint8_t m_bits = (uint8_t)(mant & 0x7);
        return s | (e_bits << 3) | m_bits;
      }
    }

    // 常规（eb=1..14）
    uint8_t e_bits = (uint8_t)(eb & 0xF);
    uint8_t m_bits = (uint8_t)(mant & 0x7);
    return s | (e_bits << 3) | m_bits;
  }

  // ---- 把 4 个 float 打包成 e4m3x4，然后拼成 32-bit ----
  __device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t* dest, float* s0, float* s1) {
    // 注意：你原 PTX 的打包顺序是
    //   lo = cvt(..., s0[1], s0[0]); hi = cvt(..., s1[1], s1[0]); mov.b32 {lo,hi}
    // 具体字节序请按你的上游/下游约定来（下面给出常用“小端：b0=LSB”示例）
    uint8_t b0 = float_to_e4m3_rn_satfinite_relaxed(s0[0]);
    uint8_t b1 = float_to_e4m3_rn_satfinite_relaxed(s0[1]);
    uint8_t b2 = float_to_e4m3_rn_satfinite_relaxed(s1[0]);
    uint8_t b3 = float_to_e4m3_rn_satfinite_relaxed(s1[1]);

    // 32-bit 小端拼接：最低字节是 b0
    *dest = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
  }

} // namespace detail

template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  // static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16

  uint32_t head_id = blockIdx.x;
  uint32_t batch_id = blockIdx.y;
  uint32_t d_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t num_threads = blockDim.x;
  uint32_t gmem_stride = num_threads * pack_size;
  // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
  uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
  uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;

  T x_val[8];
  float x_val_float[8];
  uint32_t x_val_fp8[2];

  float max_val = - 1000000.0f;
  float min_val = 1000000.0f;
  float sum_val = 0.0f;

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      float x_temp = convert_to_float(x_val[j]);
      max_val = fmaxf(max_val, x_temp);
      min_val = fminf(min_val, x_temp);

      if constexpr (sub_mean)
      {
        sum_val += x_temp;
      }
    }
  }

  // reduce
  __shared__ float s_amax_val;
  __shared__ float s_mean_val;

  float block_max_val = vllm::blockReduceMax(max_val);
  float block_min_val = vllm::blockReduceMin(min_val);
  float block_sum_val;

  if constexpr (sub_mean)
  {
    block_sum_val = vllm::blockReduceSum(sum_val);
  }

  if (thread_id == 0)
  {
    s_mean_val = block_sum_val / fp8_padded_num_tokens;

    if constexpr (sub_mean)
    {
      s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
      mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
    }
    else
    {
      s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
    }

    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
  }

  __syncthreads();

  float mean_val = s_mean_val;
  float recp_scale = scale_max / s_amax_val;

  // recalculate num_iters to cover all fp8 output tokens to prevent nan in random initialization
  uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
  num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      x_val_float[j] = convert_to_float(x_val[j]);
      if constexpr (sub_mean)
      {
        x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
      }
      else
      {
        x_val_float[j] *= recp_scale;
      }
    }

    detail::floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
    detail::floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

    detail::store_8fp8(&x_val_fp8[0], output_ptr_base + i * gmem_stride);
  }
}


template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
{

//   static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  detail::predicated_g2s_16B<pad_zero, T>(
      &shared_load[smem_load_row][ (thread_id % num_threads_per_token) * pack_size ],
      input_ptr_base,
      thread_base_token < num_tokens);
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

    *reinterpret_cast<detail::vec16_t*>(output_ptr_base) =
        *reinterpret_cast<const detail::vec16_t*>(
            &shared_store[ thread_id / num_threads_per_cta ]
                        [ (thread_id % num_threads_per_cta) * pack_size ]);
}


void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int num_tokens_padded = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.size(2);
    head_dim = input.size(1);
    stride_d_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_heads = input.size(1);
    head_dim = input.size(2);
    stride_d_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, false, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data_ptr()),
      reinterpret_cast<int8_t*>(output.data_ptr()),
      nullptr,
      reinterpret_cast<float*>(scale.data_ptr()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      0, 0,
      scale.stride(0), scale.stride(1)
    );
  });
}

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);

  constexpr int CTA_SIZE = 64;

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, padded_num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
  }

  auto input_dtype = input.scalar_type();
  auto output_dtype = output.scalar_type();

  TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

      static_assert(CTA_SIZE * HEAD_DIM <= 8192);

      dim3 block(CTA_SIZE * (HEAD_DIM / 8));

      TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data_ptr()),
        reinterpret_cast<c_type*>(output.data_ptr()),
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_d_output, stride_h_output
      );
    });
  });
}
