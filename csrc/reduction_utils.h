#pragma once
#include <cuda_runtime.h> 
#include <hip/hip_runtime.h>
#include <limits>

namespace vllm {

__device__ __forceinline__ unsigned full_mask() {
  // HIP 忽略 mask；CUDA 仍然需要一个 32-bit 掩码，但传什么都无所谓，只要活跃线程一致。
  return 0xFFFFFFFFu;
}

template <typename T>
__device__ __forceinline__ T shfl_xor(T v, int laneMask) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // HIP 提供 __shfl_xor_sync 兼容接口；width 一定要用 warpSize
  return __shfl_xor(v, laneMask, warpSize);
#else
  return __shfl_xor_sync(full_mask(), v, laneMask, warpSize);
#endif
}

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  // 从 warpSize/2 开始，保证在 32/64 两端都能全覆盖
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val += shfl_xor(val, offset);
  }
  return val;
}

template <typename T, int NUM>
__device__ __forceinline__ void warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      val[i] += shfl_xor(val[i], offset);
    }
  }
}

template<typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
  // shared 大小至少为 warpSize，避免 lane 索引越界
  __shared__ T shared[64];   // 64 >= max(warpSize) for NV(32)/AMD(64)
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  T agg = (lane < numWarps) ? shared[lane] : T(0);
  agg = warpReduceSum(agg);
  return agg;
}

template<typename T>
__device__ __forceinline__ T blockAllReduceSum(T val) {
  __shared__ T shared[64];
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  T agg = (lane < numWarps) ? shared[lane] : T(0);
  agg = warpReduceSum(agg);
  return agg;
}

template <typename T, int NUM>
__device__ __forceinline__ void blockReduceSumV2(T* val) {
  // 第二维用 warpSize+1 做 padding，减少银行冲突，且避免越界
  __shared__ T shared[NUM][65];
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; ++i) shared[i][wid] = val[i];
  }
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
    T tmp = (lane < numWarps) ? shared[i][lane] : T(0);
    val[i] = tmp;
  }
  warpReduceSumV2<T, NUM>(val);
}

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    T other = shfl_xor(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

template<typename T>
__device__ __forceinline__ T blockReduceMax(T val) {
  __shared__ T shared[64];
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  val = warpReduceMax(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  // 用极小值初始化（注意 T 类型）
  T agg = (lane < numWarps) ? shared[lane] : T(-1e20);
  agg = warpReduceMax(agg);
  return agg;
}

template<typename T>
__device__ __forceinline__ T blockAllReduceMax(T val) {
  __shared__ T shared[64];
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  val = warpReduceMax(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  T agg = (lane < numWarps) ? shared[lane] : T(-1e20);
  agg = warpReduceMax(agg);
  return agg;
}

template<typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    T other = shfl_xor(val, offset);
    val = val < other ? val : other;
  }
  return val;
}

template<typename T>
__device__ __forceinline__ T blockReduceMin(T val) {
  __shared__ T shared[64];
  const int lane = threadIdx.x & (warpSize - 1);
  const int wid  = threadIdx.x / warpSize;

  val = warpReduceMin(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  T agg = (lane < numWarps) ? shared[lane] : T(1e20);
  agg = warpReduceMin(agg);
  return agg;
}

} // namespace vllm
