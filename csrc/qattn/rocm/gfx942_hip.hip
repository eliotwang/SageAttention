#include "attn_rocm_gfx942.h"
#include "qk_gemm_hip.hip"
#include "sv_gemm_hip.hip"
#include <hip/hip_fp8.h> 
#include <c10/core/DeviceGuard.h> 

#include <hip/hip_runtime.h>
#if defined(USE_ROCM)
  #include <ATen/hip/HIPContext.h>
  #define TORCH_STREAM  at::hip::getCurrentHIPStream().stream()
  #define DEVICE_GUARD  at::hip::HIPGuard device_guard(query.device());
#else
  #include <ATen/hip/HIPContext.h>
  #define TORCH_STREAM  at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()
  #define DEVICE_GUARD  at::hip::HIPGuardMasqueradingAsCUDA device_guard(query.device());
#endif

torch::Tensor qk_int8_sv_f8_accum_f32_attn(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value); // ensure value is contiguous to prevent troubles in the kernel
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  // TODO: how to check fp8 data type?
  // CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int batch_size = query.size(0);
  const int head_dim = query.size(3);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k, stride_h_v, stride_d_v, stride_seq_o, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);

    stride_seq_q = query.stride(1);
    stride_h_q = query.stride(2);
    stride_seq_k = key.stride(1);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_d_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_o = output.stride(2);

    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    assert(value.size(1) == head_dim);
    assert(value.size(2) == num_kv_heads);
  }
  else
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);

    stride_seq_q = query.stride(2);
    stride_h_q = query.stride(1);
    stride_seq_k = key.stride(2);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_d_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_o = output.stride(1);

    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    assert(value.size(2) == head_dim);
    assert(value.size(1) == num_kv_heads);
  }
  
  int lda = qo_len;
  int ldb = kv_len;
  int ldd = kv_len; //budui
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  torch::Tensor lse = torch::empty({0});
  if (return_lse)
  {
    lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.scalar_type();

  auto t_f32 = torch::empty({batch_size, num_qo_heads, qo_len, kv_len},
                            query.options().dtype(torch::kInt32));

  int stride_bz_t = t_f32.stride(0);
  int stride_h_t = t_f32.stride(1);
  int stride_seq_t = t_f32.stride(2);

  // c10::OptionalDeviceGuard device_guard;
  // device_guard.set_device(query.device());
  hipStream_t stream = TORCH_STREAM;

  hipEvent_t ev_qk_s, ev_qk_e, ev_soft_s, ev_soft_e, ev_sv_s, ev_sv_e;
  hipEventCreate(&ev_qk_s); hipEventCreate(&ev_qk_e);
  hipEventCreate(&ev_soft_s); hipEventCreate(&ev_soft_e);
  hipEventCreate(&ev_sv_s); hipEventCreate(&ev_sv_e);

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            assert(value.size(0) == batch_size);
            assert(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K);

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);    
            }
            else
            {
              static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
            }

            //                                     smem_Q                                     smem_K                            smem_V                     smem_O
            size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
            
            auto kernel_func_qk = qk_gemm<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                        float, false, DTypeOut, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, false, false, false>;

            hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_func_qk), hipFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(128,2);

            hipEventRecord(ev_qk_s, stream);
           hipLaunchKernelGGL(( kernel_func_qk), dim3(grid), dim3(block), smem_max, 0, 
              query.data_ptr<int8_t>(), 
              key.data_ptr<int8_t>(),
              t_f32.data_ptr<int32_t>(),
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_t, stride_seq_t, stride_h_t,
              lda, ldb, ldd);
            hipEventRecord(ev_qk_e, stream);

            hipEventRecord(ev_soft_s, stream);
              //softmax
              // ---------- build logits from t_f32 (int32) → dequantize by Q/K scales → base-2 softmax ----------
              TORCH_CHECK(t_f32.is_cuda() && t_f32.scalar_type() == at::kInt, "t_f32 must be CUDA int32");

              at::Tensor logits_i32;
              {
                const std::array<int64_t,4> sizes {batch_size, num_qo_heads, qo_len, kv_len};
                if (t_f32.sizes() == at::IntArrayRef(sizes)) {
                  logits_i32 = t_f32;
                } else if (t_f32.is_contiguous()) {
                  TORCH_CHECK(t_f32.numel() == (int64_t)batch_size * num_qo_heads * qo_len * kv_len,
                              "t_f32 numel mismatch B*H*S*S");
                  logits_i32 = t_f32.view(sizes);
                } else {
                  std::array<int64_t,4> strides_t {stride_bz_t, stride_h_t, stride_seq_t, 1};
                  logits_i32 = t_f32.as_strided(sizes, strides_t);
                }
              }

              at::Tensor logits = logits_i32.to(at::kFloat);

              // ---- 反量化：按粒度把每个 (q_row, k_col) 乘以 q_scale * k_scale ----
              const auto dev = logits.device();
              const auto long_opts = at::TensorOptions().device(dev).dtype(at::kLong);

              // 计算 head 维上的 kv 组展开：把 key_scale 扩到与 num_qo_heads 对齐
              // key_scale: [B, num_kv_heads, Gk]  ->  [B, num_qo_heads, Gk]
              at::Tensor key_scale_expanded;
              {
                TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "Hq must be divisible by Hk");
                key_scale_expanded = key_scale.repeat_interleave(num_kv_groups, /*dim=*/1);
              }

              int q_group_len = (QK_QUANT_GRAN == (int)QuantGranularity::kPerWarp)
                    ? WARP_Q
                    : (WARP_Q / 8);
              int k_group_len = (QK_QUANT_GRAN == (int)QuantGranularity::kPerWarp)
                    ? WARP_K
                    : (WARP_K / 4);

              // 放在创建 q_idx / k_idx 的那段上方或直接替换原先的构造逻辑
              auto devq = query.device();
              auto idx_opts = at::TensorOptions().dtype(at::kLong).device(devq);

              auto q_idx = at::arange(qo_len, idx_opts).div(q_group_len, /*rounding_mode=*/"trunc");
              auto k_idx = at::arange(kv_len, idx_opts).div(k_group_len, /*rounding_mode=*/"trunc");

              // 然后你的 index_select 继续用即可（结果会广播到每个 token）
              at::Tensor q_row_scale = at::index_select(query_scale, /*dim=*/2, q_idx);
              at::Tensor k_col_scale = at::index_select(key_scale_expanded, /*dim=*/2, k_idx);

              at::Tensor deq_factor = q_row_scale.unsqueeze(-1) * k_col_scale.unsqueeze(-2);
              logits = logits * deq_factor;

              // ---- 乘以 1/sqrt(d)，再做 base-2 稳定 softmax ----
              const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
              logits.mul_(inv_sqrt_d);

              if (IS_CAUSAL) {
                auto neg_inf = -std::numeric_limits<float>::infinity();
                auto tri = at::triu(at::ones({qo_len, kv_len}, logits.options()), /*diagonal=*/1);
                logits = logits + tri * neg_inf;
              }

              // base-2 softmax：等价于对 (logits - max) 乘 ln2 再做 exp
              constexpr float kLn2 = 0.6931471805599453094f;
              auto m = std::get<0>(logits.max(/*dim=*/-1, /*keepdim=*/true));
              auto z = logits - m;
              auto w = at::exp(z * kLn2);                 // 2^(z) = exp(z * ln2)
              auto denom = w.sum(/*dim=*/-1, /*keepdim=*/true);
              auto probs = w / denom;

              value = value.to(at::kFloat8_e4m3fnuz).contiguous();
              at::Tensor s = probs.to(at::kFloat8_e4m3fnuz).contiguous();

              using HostF8 = c10::Float8_e4m3fnuz;
              using DevF8  = __hip_fp8_e4m3_fnuz;

              HostF8* s_host = s.data_ptr<HostF8>();
              HostF8* v_host = value.data_ptr<HostF8>();

              DevF8*  s_ptr  = reinterpret_cast<DevF8*>(s_host);
              DevF8*  v_ptr  = reinterpret_cast<DevF8*>(v_host);

              
              auto o_fp32 = at::empty_like(output, output.options().dtype(at::kFloat));
              
              TORCH_CHECK(value.dtype() == at::kFloat8_e4m3fnuz && s.dtype() == at::kFloat8_e4m3fnuz,
                          "value/s must be Float8_e4m3fnuz");
              TORCH_CHECK(value.is_contiguous() && s.is_contiguous(), "value/s must be contiguous");
              hipEventRecord(ev_soft_e, stream);

              ldb = head_dim;
              ldd = head_dim;
              auto kernel_func_sv = sv_gemm<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                        float, false, float, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, false, false, false>;

              hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_func_sv), hipFuncAttributeMaxDynamicSharedMemorySize, smem_max);

              hipEventRecord(ev_sv_s, stream);
             hipLaunchKernelGGL(( kernel_func_sv), dim3(grid), dim3(block), smem_max, 0, 
              s_ptr,
              v_ptr,
              o_fp32.data_ptr<float>(),   
              nullptr,
              nullptr,
              nullptr,
              qo_len,
              kv_len,
              num_kv_groups,
              stride_bz_t, stride_seq_t, stride_h_t,
              stride_bz_v, stride_h_v, stride_d_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              lda, ldb, ldd);
              hipEventRecord(ev_sv_e, stream);
              output.copy_(o_fp32);
              

              float ms_qk=0.f, ms_soft=0.f, ms_sv=0.f;
              hipEventElapsedTime(&ms_qk,  ev_qk_s,  ev_qk_e);
              hipEventElapsedTime(&ms_soft, ev_soft_s, ev_soft_e);
              hipEventElapsedTime(&ms_sv,   ev_sv_s,   ev_sv_e);

              // 可选：打印或存下来
              printf("QK GEMM: %.3f ms, Softmax: %.3f ms, SV GEMM: %.3f ms\n",
                    ms_qk, ms_soft, ms_sv);

              // 销毁 events
              hipEventDestroy(ev_qk_s);  hipEventDestroy(ev_qk_e);
              hipEventDestroy(ev_soft_s); hipEventDestroy(ev_soft_e);
              hipEventDestroy(ev_sv_s);   hipEventDestroy(ev_sv_e);
          });
        });
      });
    });
  });

  return lse;
}