#include "attn_rocm_gfx942.h"
#include "qk_gemm.hip"
#include "sv_gemm.hip"
#include <hip/hip_fp8.h> 
#include <c10/core/DeviceGuard.h> 

#include <hip/hip_runtime.h>
#if defined(USE_ROCM)
  #include <ATen/hip/HIPContext.h>
  #define TORCH_STREAM  at::hip::getCurrentHIPStream().stream()
  #define DEVICE_GUARD  at::hip::HIPGuard device_guard(query.device());
#else
  #include <ATen/cuda/CUDAContext.h>
  #define TORCH_STREAM  at::cuda::getCurrentCUDAStream().stream()
  #define DEVICE_GUARD  at::cuda::CUDAGuard device_guard(query.device());
#endif

#include <atomic>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <torch/serialize.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
// #include <torch/indexing.h>   // Slice

using torch::indexing::Slice;

static inline uint32_t align_up_128(uint32_t x) {
  return (x + 127u) & ~127u;
}

#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                                  \
  do {                                                                                  \
    hipError_t e = (cmd);                                                               \
    if (e != hipSuccess) {                                                              \
      printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e));         \
      abort();                                                                          \
    }                                                                                   \
  } while (0)
#endif


static inline void pad_copy_seq_lastdim_contig_HND(void* dst, const void* src,
                                                   uint32_t B, uint32_t H,
                                                   uint32_t Seq, uint32_t D,
                                                   uint32_t Seq_pad,
                                                   size_t elem_size,
                                                   hipStream_t stream) {
  const size_t bytes_src_block = (size_t)Seq * D * elem_size;
  const size_t bytes_dst_block = (size_t)Seq_pad * D * elem_size;
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t h = 0; h < H; ++h) {
      const char* src_ptr = (const char*)src + ((size_t)b * H + h) * bytes_src_block;
      char*       dst_ptr = (char*)dst       + ((size_t)b * H + h) * bytes_dst_block;
      HIP_CHECK(hipMemcpyAsync(dst_ptr, src_ptr, bytes_src_block,
                               hipMemcpyDeviceToDevice, stream));
    }
  }
}

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

  auto pad_seq128 = [](const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 4, "expect [B,H,Seq,D]");
    const int64_t seq = x.size(2);
    const int64_t pad = (128 - (seq % 128)) % 128;
    if (pad == 0) return x.contiguous();
    return at::constant_pad_nd(x, {0,0, 0,pad}, /*value=*/0).contiguous();
  };

  const int64_t B  = query.size(0);
  const int64_t Hq = query.size(1);
  const int64_t Hk = key.size(1);
  const int64_t M  = query.size(2);
  const int64_t N  = key.size(2);
  const int64_t D  = query.size(3);

  auto Q_pad = pad_seq128(query);   // [B,Hq,M_pad,D]
  auto K_pad = pad_seq128(key);     // [B,Hk,N_pad,D]
  int M_pad = (127 + M) / 128 * 128;
  int N_pad = (127 + N) / 128 * 128;
  auto T_pad = torch::empty({B, Hq, M_pad, N_pad},
                          query.options().dtype(torch::kInt32)).contiguous();
 
  const uint32_t stride_bz_q = (uint32_t)Q_pad.stride(0);
  const uint32_t stride_h_q  = (uint32_t)Q_pad.stride(1);
  const uint32_t stride_seq_q= (uint32_t)Q_pad.stride(2);

  const uint32_t stride_bz_k = (uint32_t)K_pad.stride(0);
  const uint32_t stride_h_k  = (uint32_t)K_pad.stride(1);
  const uint32_t stride_seq_k= (uint32_t)K_pad.stride(2);

  const uint32_t stride_bz_t = (uint32_t)T_pad.stride(0);
  const uint32_t stride_h_t  = (uint32_t)T_pad.stride(1);
  const uint32_t stride_seq_t= (uint32_t)T_pad.stride(2);

  const int batch_size = query.size(0);
  const int num_kv_heads = Hk;
  const int num_qo_heads = Hq;
  const int qo_len = M;
  const int kv_len = N;
  const int head_dim = D;

  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);
  const int stride_h_v = value.stride(1);
  const int  stride_d_v = value.stride(2);
  const int  stride_seq_o = output.stride(2);
  const int  stride_h_o = output.stride(1);

  CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
  CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
  assert(value.size(2) == head_dim);
  assert(value.size(1) == num_kv_heads);
  

    hipStream_t stream = TORCH_STREAM;
  //

  int lda = D;
  int ldb = D;
  int ldd = N_pad;
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

  // auto t_f32 = torch::empty({batch_size, num_qo_heads, qo_len, kv_len},
  //                           query.options().dtype(torch::kInt32));

  // int stride_bz_t = t_f32.stride(0);
  // int stride_h_t = t_f32.stride(1);
  // int stride_seq_t = t_f32.stride(2);

  // c10::OptionalDeviceGuard device_guard;
  // device_guard.set_device(query.device());
  // hipStream_t stream = TORCH_STREAM;

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
            
            auto kernel_func_qk = qk_gemm<CTA_Q, 128, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                        float, false, DTypeOut, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, false, false, false>;

            hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_func_qk), hipFuncAttributeMaxDynamicSharedMemorySize, smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(128,2);

            // hipEventRecord(ev_qk_s, stream);
            kernel_func_qk<<<grid, block, smem_max>>>(
              Q_pad.data_ptr<int8_t>(), 
              K_pad.data_ptr<int8_t>(),
              T_pad.data_ptr<int32_t>(),
              M_pad,N_pad,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_t, stride_seq_t, stride_h_t,
              lda, ldb, ldd);
            
              auto t_f32 = torch::empty({batch_size, num_qo_heads, qo_len, kv_len},
                            query.options().dtype(torch::kInt32));
              auto T_valid = T_pad.narrow(/*dim=*/2, /*start=*/0, /*length=*/M)
                        .narrow(/*dim=*/3, /*start=*/0, /*length=*/N);
              t_f32.copy_(T_valid);
              // t_f32.clamp_(-65535, 65535);

              int stride_bz_tsm = t_f32.stride(0);
              int stride_h_tsm = t_f32.stride(1);
              int stride_seq_tsm = t_f32.stride(2);
              
              Q_pad.reset();
              K_pad.reset();
              T_pad.reset();

            // hipEventRecord(ev_qk_e, stream);
            //  {
            //   static const char* kMarkerPath = "/home/tmp/rightqk/qk_first.marker";
            //   static const char* kDumpPath   = "/home/tmp/rightqk/qk_int32_first.pt";

            //   // 用“原子创建文件”做跨进程一次性保护
            //   int fd = ::open(kMarkerPath, O_CREAT | O_EXCL | O_WRONLY, 0644);
            //   if (fd >= 0) {
            //     ::write(fd, "saved\n", 6);
            //     ::close(fd);

            //     // 等 QK kernel 结束，确保 t_f32 可读
            //     hipStreamSynchronize(TORCH_STREAM);

            //     // 把 t_f32 视为 [B, H_qo, qo_len, kv_len]
            //     std::array<int64_t,4> sizes{batch_size, num_qo_heads, qo_len, kv_len};
            //     at::Tensor logits_i32;
            //     if (t_f32.sizes() == at::IntArrayRef(sizes)) {
            //       logits_i32 = t_f32;
            //     } else if (t_f32.is_contiguous()) {
            //       logits_i32 = t_f32.view(sizes);
            //     } else {
            //       std::array<int64_t,4> strides_t{stride_bz_tsm, stride_h_tsm, stride_seq_tsm, 1};
            //       logits_i32 = t_f32.as_strided(sizes, strides_t);
            //     }

            //     // 拷到 CPU 后保存（整张矩阵）
            //     auto cpu_t = logits_i32.to(at::kCPU).contiguous();
            //     try {
            //       torch::save(cpu_t, kDumpPath);
            //       printf("[QK-DUMP] saved full int32 QK (t) to %s\n", kDumpPath);
            //     } catch (const std::exception& e) {
            //       fprintf(stderr, "[QK-DUMP] torch::save failed: %s\n", e.what());
            //       // 兜底：保存为原始二进制
            //       std::ofstream ofs("/home/SageAttention/qk_int32_first.bin", std::ios::binary);
            //       ofs.write(reinterpret_cast<const char*>(cpu_t.data_ptr<int32_t>()),
            //                 cpu_t.numel() * sizeof(int32_t));
            //       ofs.close();
            //       printf("[QK-DUMP] wrote raw .bin fallback to /home/SageAttention/qk_int32_first.bin\n");
            //     }
            //   }
            //   // 若 fd<0，说明别的进程/此前调用已保存过，直接跳过
            // }

            // hipEventRecord(ev_soft_s, stream);
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
                  std::array<int64_t,4> strides_t {stride_bz_tsm, stride_h_tsm, stride_seq_tsm, 1};
                  logits_i32 = t_f32.as_strided(sizes, strides_t);
                }
              }

              at::Tensor logits = logits_i32.to(at::kFloat);

              // const auto dev = logits.device();
              // const auto long_opts = at::TensorOptions().device(dev).dtype(at::kLong);

              const int64_t Q_TILES_PER_SEQ = query_scale.size(2);
              const int64_t K_TILES_PER_SEQ = key_scale.size(2);

              auto dev  = query.device();

              constexpr int64_t Q_WARP   = 32;   // WARPQ
              constexpr int64_t K_WARP   = 64;   // WARPK
              constexpr int64_t Q_SLICES = 8;    // 每 WARP 的 threadlet 数（Q）
              constexpr int64_t K_SLICES = 4;    // 每 WARP 的 threadlet 组（K）
              TORCH_CHECK((8 % K_SLICES) == 0, "K_SLICES must divide 8");
              const int64_t step_in8 = 8 / K_SLICES;

              auto idxopts = at::TensorOptions().dtype(at::kLong).device(query.device());
              auto offs_m = at::arange(qo_len, idxopts);  // [0..qo_len)
              auto offs_n = at::arange(kv_len, idxopts);  // [0..kv_len)

              
              // —— Q 侧：全程整型 ——
              // row_blk = floor_div(offs_m, Q_WARP)
              auto row_blk   = at::floor_divide(offs_m, Q_WARP);
              // row_slice = (offs_m % Q_WARP) % Q_SLICES
              auto row_slice = at::remainder(offs_m, Q_WARP);
              row_slice      = at::remainder(row_slice, Q_SLICES);
              // q_idx = row_blk * Q_SLICES + row_slice
              auto q_idx     = row_blk * Q_SLICES + row_slice;          // Long

              // —— K 侧：全程整型 ——
              // k_blk = floor_div(offs_n, K_WARP)
              auto k_blk   = at::floor_divide(offs_n, K_WARP);
              // grp_in8 = floor_div(offs_n % 8, step_in8)
              auto grp_in8 = at::remainder(offs_n, 8);
              grp_in8      = at::floor_divide(grp_in8, step_in8);
              // k_idx = k_blk * K_SLICES + grp_in8
              auto k_idx   = k_blk * K_SLICES + grp_in8;                // Long

              // （可选）保险起见，确保 Long、同设备、连续
              q_idx = q_idx.to(at::kLong).contiguous();
              k_idx = k_idx.to(at::kLong).contiguous();

              TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "Hq % Hk != 0");
              auto key_scale_expanded = key_scale.repeat_interleave(num_qo_heads / num_kv_heads, /*dim=*/1);

              // 按索引 gather（与 Triton 完全同构）
              auto q_row_scale = at::index_select(query_scale,        /*dim=*/2, q_idx); // [B,Hq,qo_len]
              auto k_col_scale = at::index_select(key_scale_expanded, /*dim=*/2, k_idx); // [B,Hq,kv_len]

              // 反量化
              logits = logits * q_row_scale.unsqueeze(-1) * k_col_scale.unsqueeze(-2);

              const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
              logits.mul_(inv_sqrt_d);

              if (IS_CAUSAL) {
                auto neg_inf = -std::numeric_limits<float>::infinity();
                auto mask = at::triu(
                  at::ones({qo_len, kv_len}, at::TensorOptions().device(logits.device()).dtype(at::kBool)),
                    /*diagonal=*/1);
                logits = logits.masked_fill(mask, neg_inf);
              }

              // base-2 softmax：
              // 1) 先清洗：非有限一律视为 -Inf（防止 max 被 NaN 污染）
              auto logits_clean = at::where(
                  at::isfinite(logits),
                  logits,
                  at::full_like(logits, -std::numeric_limits<float>::infinity())
              );

              // 2) 再做 padding / causal 等 mask（至少保证每行有一个有效列为 true）
              const auto idxoptsa = at::TensorOptions().dtype(at::kLong).device(dev);
              const int64_t KV = logits_clean.size(-1);  // 实际本轮长度（8866）

              if (kv_len < KV) {  // 只有当 logits 含 padding 时才盖
                  auto pad_mask = at::arange(KV, idxoptsa) >= kv_len;  // [KV]
                  logits_clean = logits_clean.masked_fill(
                      pad_mask.view({1,1,1,KV}).expand_as(logits_clean),
                      -std::numeric_limits<float>::infinity()
                  );
              }

              // 3) 现在才做稳定 softmax（基 2）：
              auto m = std::get<0>(logits_clean.max(/*dim=*/-1, /*keepdim=*/true));
              auto z = logits_clean - m;              // 不会再被 NaN 污染
              auto w = at::exp2(z);                   // 2^z
              auto denom = w.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-20f);
              auto probs = w / denom;

              // 兜底：极端情况下某行全被 mask（理论上不该发生），给有效列均匀分布
              auto row_sum = probs.sum(-1, /*keepdim=*/true);
              auto need_uniform = row_sum.eq(0);
              if (need_uniform.any().item<bool>()) {
                  // 构造 valid_mask：有效列为 1，无效列为 0（按你的布局生成）
                  auto idx = at::arange(N_pad, idxopts);
                  auto valid_mask = (idx < kv_len).view({1,1,1,-1}).to(probs.dtype()).expand_as(probs);
                  auto uniform = valid_mask / valid_mask.sum(-1, true).clamp_min(1e-20);
                  probs = at::where(need_uniform, uniform, probs);
              }

              value = value.to(at::kFloat8_e4m3fnuz).contiguous();

              
              // // ====== 量化前（probs: float32/float16 softmax 输出）======
              // auto probs_cpu = probs.to(torch::kFloat32).to(torch::kCPU).contiguous();
              // auto nz_count  = at::count_nonzero(probs_cpu).item<int64_t>();
              // double nz_frac = static_cast<double>(nz_count) / static_cast<double>(probs_cpu.numel());

              // // 最小正数（>0）
              // auto pos_mask  = probs_cpu > 0;
              // double min_pos = 0.0;
              // if (pos_mask.any().item<bool>()) {
              //   min_pos = probs_cpu.masked_select(pos_mask).min().item<float>();
              // }

              // std::cout << std::fixed << std::setprecision(6);
              // std::cout << "[probs] nonzero frac: " << nz_frac
              //           << ", nonzero count: " << nz_count
              //           << ", min>0: " << min_pos << std::endl;
              

              HIP_CHECK(hipStreamSynchronize(TORCH_STREAM));
              int k = std::max(0, (int)std::ceil(std::log2((double)N_pad) - 6.0));
              float c = (float)(1u << k);
              auto S_scaled = probs * c;
              at::Tensor s = S_scaled.to(at::kFloat8_e4m3fnuz).contiguous();
              

              // // ====== 量化后（s: Float8_e4m3fnuz）======
              // auto s_cpu   = s.to(torch::kFloat32).to(torch::kCPU).contiguous();
              // auto nz8_cnt = at::count_nonzero(s_cpu).item<int64_t>();
              // double nz8_frac = static_cast<double>(nz8_cnt) / static_cast<double>(s_cpu.numel());

              // auto pos_mask8 = s_cpu > 0;
              // double min_pos8 = 0.0;
              // if (pos_mask8.any().item<bool>()) {
              //   min_pos8 = s_cpu.masked_select(pos_mask8).min().item<float>();
              // }

              // std::cout << "[s(fp8->f32)] nonzero frac: " << nz8_frac
              //           << ", nonzero count: " << nz8_cnt
              //           << ", min>0: " << min_pos8 << std::endl;

              // {
              //   static const char* kMarkerPath = "/home/tmp/rightqk/s_first.marker";
              //   int fd = ::open(kMarkerPath, O_CREAT | O_EXCL | O_WRONLY, 0644);
              //   if (fd >= 0) {
              //     ::write(fd, "saved\n", 6);
              //     ::close(fd);

              //     hipStreamSynchronize(TORCH_STREAM);  // 确保 logits/probs/s 已就绪

              //     // 1) 反量化+1/sqrt(d) 之后的 logits（float32）
              //     try {
              //       auto logits_cpu = logits.to(at::kCPU).contiguous();  // [B,H,qo_len,kv_len], float
              //       torch::save(logits_cpu, "/home/tmp/rightqk/qk_logits_deq_first.pt");
              //       printf("[QK-DUMP] saved dequantized logits to /home/tmp/qk_files/qk_logits_deq_first.pt\n");
              //     } catch (const std::exception& e) {
              //       fprintf(stderr, "[QK-DUMP] save logits failed: %s\n", e.what());
              //     }

              //     // 2) softmax 后的概率（float32）
              //     try {
              //       auto probs_cpu = p.to(at::kCPU).contiguous();    // [B,H,qo_len,kv_len], float
              //       torch::save(probs_cpu, "/home/tmp/rightqk/qk_probs_first.pt");
              //       printf("[QK-DUMP] saved softmax probs to /home/tmp/qk_files/qk_probs_first.pt\n");
              //     } catch (const std::exception& e) {
              //       fprintf(stderr, "[QK-DUMP] save probs failed: %s\n", e.what());
              //     }

              //     // 3) 概率的 FP8 量化版本（可选，方便核对你的 SV 路径输入）
              //     try {
              //       auto s_cpu = s.to(at::kCPU).contiguous();            // [B,H,qo_len,kv_len], float8_e4m3fnuz
              //       torch::save(s_cpu, "/home/tmp/rightqk/qk_probs_fp8_first.pt");
              //       printf("[QK-DUMP] saved probs(FP8) to /home/tmp/qk_files/qk_probs_fp8_first.pt\n");
              //     } catch (const std::exception& e) {
              //       fprintf(stderr, "[QK-DUMP] save probs(FP8) failed: %s\n", e.what());
              //       // 兜底：转回 float 再存
              //       try {
              //         auto s_as_f32_cpu = s.to(at::kFloat).to(at::kCPU).contiguous();
              //         torch::save(s_as_f32_cpu, "/home/tmp/qk_files/qk_probs_fp8_as_float_first.pt");
              //         printf("[QK-DUMP] saved probs(FP8->float) to /home/tmp/qk_files/qk_probs_fp8_as_float_first.pt\n");
              //       } catch (const std::exception& e2) {
              //         fprintf(stderr, "[QK-DUMP] fallback save probs(FP8->float) failed: %s\n", e2.what());
              //       }
              //     }
              //   }
              //   // 若 fd<0，说明之前已保存过，跳过
              // }

              auto T_pad = at::constant_pad_nd(
                s.contiguous(),                           
                { (int64_t)(N_pad - N), 0, (int64_t)(M_pad - M), 0 },  // {N_right,N_left, M_right,M_left}
                0
              ).contiguous();

              auto V_pad = at::constant_pad_nd(
                value.contiguous(),                      
                { (int64_t)(N_pad - value.size(3)), 0 },            
                0
              ); 
              V_pad = V_pad.permute({0,1,3,2}).contiguous();
              auto O_pad = at::zeros({B, Hq, M_pad, D}, output.options().dtype(torch::kFloat)).contiguous();
              
              const uint32_t stride_bz_t = (uint32_t)T_pad.stride(0);  // = Hq * M_pad * N_pad
              const uint32_t stride_h_t  = (uint32_t)T_pad.stride(1);  // = M_pad * N_pad
              const uint32_t stride_seq_t= (uint32_t)T_pad.stride(2);  // = N_pad

              const uint32_t stride_bz_v = (uint32_t)V_pad.stride(0);
              const uint32_t stride_h_v  = (uint32_t)V_pad.stride(1);
              const uint32_t stride_seq_v  = (uint32_t)V_pad.stride(2);

              const uint32_t stride_bz_o = (uint32_t)O_pad.stride(0);
              const uint32_t stride_h_o  = (uint32_t)O_pad.stride(1);
              const uint32_t stride_seq_o= (uint32_t)O_pad.stride(2);

              const uint32_t lda = (uint32_t)N_pad;
              const uint32_t ldb = (uint32_t)D;
              const uint32_t ldd = (uint32_t)D;

              using HostF8 = c10::Float8_e4m3fnuz;
              using DevF8  = __hip_fp8_e4m3_fnuz;
              
              TORCH_CHECK(value.dtype() == at::kFloat8_e4m3fnuz && s.dtype() == at::kFloat8_e4m3fnuz,
                          "value/s must be Float8_e4m3fnuz");
              TORCH_CHECK(value.is_contiguous() && s.is_contiguous(), "value/s must be contiguous");
            //   hipEventRecord(ev_soft_e, stream);

              
              auto kernel_func_sv = sv_gemm<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                        float, false, float, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, false, false, false>;

              hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_func_sv), hipFuncAttributeMaxDynamicSharedMemorySize, smem_max);

              // lda = kv_len;
              // ldb = kv_len;
              // ldd = head_dim;
            //   hipEventRecord(ev_sv_s, stream);
              kernel_func_sv<<<grid, block, smem_max>>>(
              reinterpret_cast<DevF8*>(T_pad.data_ptr<HostF8>()),
              reinterpret_cast<DevF8*>(V_pad.data_ptr<HostF8>()),
              O_pad.data_ptr<float>(),   
              nullptr,
              nullptr,
              nullptr,
              (uint32_t)M_pad, (uint32_t)N_pad,
              num_kv_groups,
              stride_bz_t, stride_seq_t, stride_h_t,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              lda, ldb, ldd);
              // hipEventRecord(ev_sv_e, stream);
              
              HIP_CHECK(hipGetLastError());
              HIP_CHECK(hipStreamSynchronize(stream));
              
              O_pad.div_(c);                            // 补偿全局放大
              auto O_valid = O_pad.narrow(/*dim=*/2, 0, M);    // [B,H,M,D]
              O_valid.mul_(value_scale.unsqueeze(2));       // 右乘列缩放（per-D） 
              output.copy_(O_valid);
              
              // auto P_ref = probs;
              // auto V_ref = value.to(at::kFloat);
              // V_ref.mul_(value_scale.unsqueeze(-1));

              // V_ref = V_ref.permute({0,1,3,2}).contiguous();
              // // TORCH_CHECK(P_ref.size(-1) == V_ref.size(-2));
              // std::cout << "p_ref.size(-1):"<<P_ref.size(-1)<<"v_ref.size(-2):"<<
              // V_ref.size(-2)<<std::endl;
              // auto O_ref = at::matmul(P_ref, V_ref);

              // auto O_ref_head = O_ref[0][0].contiguous().view(-1);
              // auto O_ker_head = O_pad[0][0].contiguous().view(-1);
              // std::cout << "[cmp] ref max=" << O_ref_head.abs().max().item<float>()
              //           << " ker max=" << O_ker_head.abs().max().item<float>() << "\n";
              // double l2 = (O_ref_head - O_ker_head).norm().item<double>() /
              //             (O_ref_head.norm().item<double>() + 1e-6);
              // std::cout << "[cmp] relative L2=" << l2 << "\n";
            {
              static const char* kMarkerPath = "/home/tmp/o_files/o_first.marker";
              static const char* kDumpPath   = "/home/tmp/o_files/o_first.pt";

              int fd = ::open(kMarkerPath, O_CREAT | O_EXCL | O_WRONLY, 0644);
              if (fd >= 0) {
                ::write(fd, "saved\n", 6);
                ::close(fd);

                hipStreamSynchronize(TORCH_STREAM);  // 确保 o_fp32 已经写完

                auto o_cpu = output.to(at::kCPU).contiguous();
                try {
                  torch::save(o_cpu, kDumpPath);
                  printf("[O-DUMP] saved first O tensor to %s\n", kDumpPath);
                } catch (const std::exception& e) {
                  fprintf(stderr, "[O-DUMP] save failed: %s\n", e.what());
                  // 兜底写成二进制
                  std::ofstream ofs("/home/SageAttention/o_first.bin", std::ios::binary);
                  ofs.write(reinterpret_cast<const char*>(o_cpu.data_ptr<float>()),
                            o_cpu.numel() * sizeof(float));
                  ofs.close();
                  printf("[O-DUMP] wrote raw .bin fallback to /home/SageAttention/o_first.bin\n");
                }
              }
            }

            //   float ms_qk=0.f, ms_soft=0.f, ms_sv=0.f;
            //   hipEventElapsedTime(&ms_qk,  ev_qk_s,  ev_qk_e);
            //   hipEventElapsedTime(&ms_soft, ev_soft_s, ev_soft_e);
            //   hipEventElapsedTime(&ms_sv,   ev_sv_s,   ev_sv_e);

              // 可选：打印或存下来
            //   printf("QK GEMM: %.3f ms, Softmax: %.3f ms, SV GEMM: %.3f ms\n",
            //         ms_qk, ms_soft, ms_sv);

              // 销毁 events
            //   hipEventDestroy(ev_qk_s);  hipEventDestroy(ev_qk_e);
            //   hipEventDestroy(ev_soft_s); hipEventDestroy(ev_soft_e);
            //   hipEventDestroy(ev_sv_s);   hipEventDestroy(ev_sv_e);
          });
        });
      });
    });
  });

  return lse;
}