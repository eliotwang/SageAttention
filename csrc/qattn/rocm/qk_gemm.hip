/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "attn_utils.h"
#include "common.hpp"

/* Motivation
*
* For this particular GEMM kernel, high performance can be
* achieved through two general principles:
* 1) Data re-use
* 2) Latency hiding
*
* From the simple_gemm implementation, we know that the GEMM
* equation takes the form:
*
* D = alpha * AxB + beta * C, where
*
* A, B = input tiles of MxK and KxN, respectively
* C = input tile of MxN and
* D = final output tile, MxN
* alpha, beta are scalar factors
* (M, N and K are block dimensions)
*
* In the simple_gemm sample, each warp is responsible for computing
* one output D tile of the final result. In the current sample, each
* warp is now responsible for computing multiple D tiles, what we
* might call a Warp Tile. Because Warp Tile blocks share data locality
* in either the same row or column direction, warps can re-use input
* data from A and B as they step through the K dimension for each block.
*
* Moreover, Warp Tiles processed by warps in a thread block
* have common locality in the larger Macro Tile. In the Global D layout
* shown below, data re-use opportunities await in D tiles aligned in the
* same rows / columns. These will pass over the same input A/B values as
* they march through the K dimension.
*
* Block size:      (BlockM x BlockN)
* Warp tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
* Macro Tile size: (TBlock.x * WarpTileSize.x) x (TBlock.y * WarpTileSize.y)
*
* Wave data share input A: same row
* Wave data share input B: same col
*
* Global D layout & warp assignment for BlocksX = BlocksY = 2, 2x2 Warps
*
* W (X, Y) = wave row X, col Y
*                                     |--------- Macro Tile Y-------------|
*                                     |-- Wave Tile Y --|
*                                     |-BlockN-|
*
*                                      BlockN x BlocksY   BlockN x BlocksY
*                                     |<--------------->|<--------------->|
*      _ _   _ _      _ _          ___  ________ ________ ________ ________
*       |     |        |            ^  |        |        |        |        |
*       | Wave| BlockM |   BlockM   |  |        W        |        W        |
*       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
*       |  X  |            BlocksX  |  |                 |                 |
* Macro |     |                     |  |                 |                 |
*  Tile |    _|_                   _v_ |________|________|________|________|
*   X   |                           ^  |        |        |        |        |
*       |                  BlockM   |  |        W        |        W        |
*       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
*       |                  BlocksX  |  |                 |                 |
*       |                           |  |                 |                 |
*      _|_                         _v_ |________|________|________|________|
*
*
* From the above diagram, we can see that input A/B data can be shared within warps,
* as well as between warps in the same threadblock. This means that warps in the same
* thread block can share the input loading responsibilities if they synchronize stepping
* through the K dimension for tiles at the same time.
*
* rocWMMA Cooperative API allows thread blocks to collaboratively move data from
* one location to another. In this case, we will move data from global memory space to
* local storage such that inter-warp data sharing is possible. Maximizing data re-use
* in this way reduces costly access to global memory and improves performance.
*
* To maximize efficiency, we can structure the kernel to maximize bandwidth usage and keep
* the compute resources as busy as possible at the same time. Using a pre-fetch technique,
* we can fetch A/B inputs for the next K-step while keeping the compute resources busy
* processing the current K-step. This helps to hide memory fetching latency.
*
* In general, the process would flow like the following:
*
*       Start
*         |
*   Pre-Fetch Global A/B for K0
*         |
*   Store LDS buffer0
*         |
*         v
*   Loop: i = 1:K-1
*   ^         |
*   |    Fetch Global A/B i+1; store LDS Buffer 1
*   |         |
*   |    Load LDS buffer0; Accum A x B
*   |         |
*   |    Swap buffer0, buffer1
*   |         |
*   |         |
*   end_loop <-
*         |
*   Load LDS buffer0; Accum A x B
*         |
*   Load Global C Tile
*         |
*   D = alpha * AccumAB + beta * C
*         |
*   Write D Tile
*         |
*         v
*        End
*
* Lds Mapping
* Buffer Width = LDS Width = BlockK
* Matrix geometry for inputs A and B have a common dimension (BlockK).
* We can fix one of the LDS dimensions to BlockK (in this case the width),
* and insert blocks of different heights (BlockM, BlockN) to use the space
* without the need of extra padding.
*
* Fragments of B must be transposed to fit this geometry,
* and both fragments from A and B must accomodate LDS data layout.
*
* Local Layout (LDS):
*
* Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
* of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
*
* Transposed B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
* Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
*
*
*                        _____________BlockK_____________
*                       |                                |
*                       v                                v
*                  (0,0) ----------------------------------->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockM |   | C0 | C1 | C2               | Ck-1 |   A0
*  Tile X  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...          AX-1
*          -->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
*  Tile Y  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...        BY-1 (T)
*          -->                                           (MacroTileX + MacroTileY - 1, BlockK -1)
*
* Depending on the locality of the block being processed, warps load the corresponding
* A and B inputs from LDS buffer and use them for the accumulation of AxB calculations.
*/

using namespace rocwmma;

///
/// Parameter configuration
///

/* Depending on the GPU architecture this sample is run on, the following kernel parameters need to
*  be modified in order to obtain high performance.
* _________________________________________________________________________________________
*|         |           |           |           |          |          |          |          |
*|         | ROCWMMA_M | ROCWMMA_N | ROCWMMA_K | BLOCKS_X | BLOCKS_Y | TBLOCK_X | TBLOCK_Y |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_9  |    32     |    32     |    16     |    2     |    2     |   128    |    2     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_11 |    16     |    16     |    16     |    4     |    2     |    64    |    4     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*
* __________________________________________
*|         |                                |
*|         |           WARP_SIZE            |
*|_________|________________________________|
*|         |                                |
*|  GFX_9  | Constants::AMDGCN_WAVE_SIZE_64 |
*|_________|________________________________|
*|         |                                |
*|  GFX_11 | Constants::AMDGCN_WAVE_SIZE_32 |
*|_________|________________________________|
*/




///
/// Types and Data Layouts
///

using InputTQK   = int8_t;
using OutputTQK  = int32_t;
using ComputeTQK = int32_t;


using DataLayoutQ   = row_major;
using DataLayoutK   = col_major;
using DataLayoutQK   = row_major;
using DataLayoutQKLds = col_major;

///
/// Fragment types
///



// Mfma frags
using MfmaFragQ   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputTQK, DataLayoutQ>;
using MfmaFragK   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputTQK, DataLayoutK>;
using MfmaFragQKS   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputTQK, DataLayoutQK>;
using MfmaFragQKD   = MfmaFragQKS;
using MfmaFragQKAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeTQK, DataLayoutQK>;

// Global read (macro tile)
using GRBuffQ = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputTQK, DataLayoutQ>;
using GRBuffK = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputTQK, DataLayoutK>;

// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWBuffQ = ApplyDataLayout_t<GRBuffQ, DataLayoutQKLds>;
using LWBuffK = ApplyDataLayout_t<ApplyTranspose_t<GRBuffK>, DataLayoutQKLds>;

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragQ = ApplyDataLayout_t<MfmaFragQ, DataLayoutQKLds>;
using LRFragK = ApplyDataLayout_t<ApplyTranspose_t<MfmaFragK>, DataLayoutQKLds>;
// #endif // (ROCWMMA_ARCH_GFX9 || ROCWMMA_ARCH_GFX11)

///
/// Wrapper functions: repeat mfma tile operations across entire warp tile.
///

// Cooperative global read / local write (Macro tile data movement)
// Loads / stores a global data fragment cooperatively across warps. Each participating warp is
// responsible for only a portion of the whole fragment.
//
// The cooperative operation is split into work items (SplitCount). Work items are consumed in
// a round robin fashion by warps in the range of [0, WaveCount). The wave index determines the
// order of the current wave in the collaboration pool.
//
// WaveCount, SplitCount and waveIndex parameters must match successive coop load / store calls
// to ensure the entire fragment remains coherent.

// Global A reads in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    globalReadCoopA(GRBuffQ& grBuffQ, InputTQK const* gAddrA, uint32_t lda, uint32_t waveIndexA)
{
    load_matrix_coop_sync<WaveCountA>(grBuffQ, gAddrA, lda, waveIndexA);
}

// Global B reads in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    globalReadCoopB(GRBuffK& grBuffK, InputTQK const* gAddrB, uint32_t ldb, uint32_t waveIndexB)
{
    load_matrix_coop_sync<WaveCountB>(grBuffK, gAddrB, ldb, waveIndexB);
}

// Local A writes in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputTQK* ldsAddr, GRBuffQ const& grBuffQ, uint32_t ldsld, uint32_t waveIndexA)
{
    // No transpose, but apply the lds data layout
    store_matrix_coop_sync<WaveCountA>(
        ldsAddr, applyDataLayout<DataLayoutQKLds, WaveCountA>(grBuffQ), ldsld, waveIndexA);
}

// Local B writes in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputTQK* ldsAddr, GRBuffK const& grBuffK, uint32_t ldsld, uint32_t waveIndexB)
{
    // Transpose B and then apply lds data layout
    store_matrix_coop_sync<WaveCountB>(
        ldsAddr, applyDataLayout<DataLayoutQKLds, WaveCountB>(applyTranspose(grBuffK)), ldsld, waveIndexB);
}

// Local A reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadA(MfmaFragQ (&fragsA)[BLOCKS_X], InputTQK const* ldsAddrA, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragQ>;
    using Mapper1d  = GetDataLayout_t<LRFragQ>;

    // Each A block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragQ tmp;
        load_matrix_sync(tmp, ldsAddrA, ldsld);
        fragsA[i] = applyDataLayout<DataLayoutQ>(tmp);

        ldsAddrA += blockStep;
    }
}

// Local B reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadB(MfmaFragK (&fragsB)[BLOCKS_Y], InputTQK const* ldsAddrB, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragK>;
    using Mapper1d  = GetDataLayout_t<LRFragK>;

    // Each B block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragK tmp;
        load_matrix_sync(tmp, ldsAddrB, ldsld);

        // Transform back to MFMA tile
        fragsB[i] = applyDataLayout<DataLayoutK>(applyTranspose(tmp));

        ldsAddrB += blockStep;
    }
}


// Global D reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    globalWriteD(OutputTQK* gAddrD, MfmaFragQKD const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using FragShape = GetIOShape_t<MfmaFragQKD>;
    using Mapper1d  = GetDataLayout_t<MfmaFragQKD>;

    // Iterative offsets for each D block in the warp tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrD + offsetY, fragsD[i][j], ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

// Broadcast value to fragments in warp tile
template <typename FragT>
ROCWMMA_DEVICE static inline void fill(FragT (&frags)[BLOCKS_X][BLOCKS_Y],
                                       GetDataType_t<FragT> value)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            fill_fragment(frags[i][j], value);
        }
    }
}

// Performs warp tile mfma
ROCWMMA_DEVICE static inline void mfma(MfmaFragQKAcc (&fragsAccOut)[BLOCKS_X][BLOCKS_Y],
                                       MfmaFragQ const (&fragsA)[BLOCKS_X],
                                       MfmaFragK const (&fragsB)[BLOCKS_Y],
                                       MfmaFragQKAcc const (&fragsAccIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            mma_sync(fragsAccOut[i][j], fragsA[i], fragsB[j], fragsAccIn[i][j]);
        }
    }
}


template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool fuse_v_mean=false, bool use_pv_fp16_accu=false>
ROCWMMA_KERNEL void __launch_bounds__(256) qk_gemm(int8_t *__restrict__ Q, int8_t *__restrict__ K,int32_t *__restrict__ T, 
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q, 
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_t, const uint32_t stride_seq_t, const uint32_t stride_h_t,
                      const uint32_t lda, const uint32_t ldb, const uint32_t ldd)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        ///
        /// 2D matrix coordinate setup
        ///

        // Tile Sizes
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y);  //(64,64)
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);//(128,128)

        // Local warp coordinate relative to current threadblock (wg).
        constexpr auto warpDims        = make_coord2d(WARPS_X, WARPS_Y);        //(2,2)
        auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto           localWarpOffset = localWarpCoord * warpTileSize;


        const uint32_t batch_id = blockIdx.z;
        const uint32_t bx = blockIdx.x;
        const uint32_t num_qo_heads = gridDim.y;
        const uint32_t head_id = blockIdx.y;

        const uint32_t num_iterations = div_ceil(
            kv_len,
            MACRO_TILE_Y);

        auto outptr = T + batch_id * stride_bz_t + head_id * stride_h_t;
        // T += batch_id * stride_bz_t + head_id * stride_h_t;
        for (uint32_t iter = 0; iter < num_iterations ; iter++)
        {
            auto macroTileCoord   = make_coord2d(bx, iter) * macroTileSize;
            auto warpTileCoord  = macroTileCoord + localWarpOffset;

            // Bounds check
            auto warpTileBound = warpTileCoord + warpTileSize;
            if(get<0>(warpTileBound) > qo_len || get<1>(warpTileBound) > kv_len)
            {
                continue;
            }
            ///
            /// 1D global read coordinate setup
            ///
            using GRBuffQMap1d = GetDataLayout_t<GRBuffQ>;
            using GRBuffKMap1d = GetDataLayout_t<GRBuffK>;

            const uint32_t m0 = get<0>(macroTileCoord);
            const uint32_t n0 = get<1>(macroTileCoord);
            // Initial globa read address offsets
            auto globalReadOffsetA
                = batch_id * stride_bz_q + head_id * stride_h_q +
                GRBuffQMap1d::fromMatrixCoord(make_coord2d(m0, 0u), lda);
            auto globalReadOffsetB
                = batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + 
                GRBuffKMap1d::fromMatrixCoord(make_coord2d(0u, n0), ldb);

            // Incremental global read address offsets
            auto kStepOffsetA = GRBuffQMap1d::fromMatrixCoord(make_coord2d(0u, ROCWMMA_K), lda);
            auto kStepOffsetB = GRBuffKMap1d::fromMatrixCoord(make_coord2d(ROCWMMA_K, 0u), ldb);

            ///
            /// Cooperative config for global read A / B
            ///

            // WorkItems will be split up by minimum IOCount to perform either global read or local write.
            // These are inputs to cooperative functions.
            constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);

            // Scheduling warp order is analogous to row major priority.
            // E.g. Wg = (128, 2) = 2x2 warps
            // (0, 0)   (0, 1)   Share Schedule: w0 = (0, 0), w1 = (0, 1),
            // (1, 0)   (1, 1)                   w2 = (1, 0), w3 = (1, 1), count = 4
            const auto warpIndex = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

            ///
            /// Perform initial global pre-fetch
            ///

            GRBuffQ grBuffQ;
            GRBuffK grBuffK;

            globalReadCoopA<warpCount>(grBuffQ, Q + globalReadOffsetA, lda, warpIndex);
            globalReadCoopB<warpCount>(grBuffK, K + globalReadOffsetB, ldb, warpIndex);

            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            ///
            /// Setup LDS addressing
            /// This kernel will use 2 separate LDS blocks for pipelining
            /// the input prefetching during the accumulation loop
            ///

            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            using LWBuffQShape = GetIOShape_t<LWBuffQ>;
            using LWBuffKShape = GetIOShape_t<LWBuffK>;
            using LWBuffQMap1d = GetDataLayout_t<LWBuffQ>;
            using LWBuffKMap1d = GetDataLayout_t<LWBuffK>;

            constexpr uint32_t ldsWidth  = ROCWMMA_K;
            constexpr uint32_t ldsHeight = LWBuffQShape::BlockHeight + LWBuffKShape::BlockHeight;
            constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
            constexpr uint32_t ldsld = std::is_same_v<DataLayoutQKLds, row_major> ? ldsWidth : ldsHeight;

            auto* ldsPtrLo = reinterpret_cast<InputTQK*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + sizeLds;

            // Local write offsets to start of A / B data
            auto ldsWriteOffsetA = 0u;
            auto ldsWriteOffsetB
                = LWBuffQMap1d::fromMatrixCoord(make_coord2d(LWBuffQShape::BlockHeight, 0u), ldsld);

            // Local read offsets for mfma frags
            auto ldsReadOffsetA
                = ldsWriteOffsetA
                + LWBuffQMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
            auto ldsReadOffsetB
                = ldsWriteOffsetB
                + LWBuffKMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

            ///
            /// Write prefetch to local
            ///
            localWriteCoopA<warpCount>(ldsPtrLo + ldsWriteOffsetA, grBuffQ, ldsld, warpIndex);
            localWriteCoopB<warpCount>(ldsPtrLo + ldsWriteOffsetB, grBuffK, ldsld, warpIndex);

            ///
            /// Initialize accumulation frags
            ///
            MfmaFragQKAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
            fill(fragsAcc, 0.0f);

            ///
            /// Synchronize warps and memory
            ///
            synchronize_workgroup();

            ///
            /// Accumulate A * B for all mfma frags in warp tile
            ///
            for(uint32_t currentK = ROCWMMA_K; currentK < head_dim; currentK += ROCWMMA_K)
            {
                MfmaFragQ fragsA[BLOCKS_X];
                MfmaFragK fragsB[BLOCKS_Y];

                // Local read mfma frags from first LDS buffer
                localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
                localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

                // Prefetch next round of global frags
                globalReadCoopA<warpCount>(grBuffQ, Q + globalReadOffsetA, lda, warpIndex);
                globalReadCoopB<warpCount>(grBuffK, K + globalReadOffsetB, ldb, warpIndex);

                // Advance offsets to next k step
                globalReadOffsetA += kStepOffsetA;
                globalReadOffsetB += kStepOffsetB;

                // accum(A * B)
                mfma(fragsAcc, fragsA, fragsB, fragsAcc);

                // Write prefetch to second LDS buffer
                localWriteCoopA<warpCount>(ldsPtrHi + ldsWriteOffsetA, grBuffQ, ldsld, warpIndex);
                localWriteCoopB<warpCount>(ldsPtrHi + ldsWriteOffsetB, grBuffK, ldsld, warpIndex);

                // Make sure that all waves have finished reading / writing to lds for currentK.
                synchronize_workgroup();

                // Swap Lds buffers
                auto* tmp = ldsPtrLo;
                ldsPtrLo  = ldsPtrHi;
                ldsPtrHi  = tmp;
            }

            ///
            /// Start loading C
            ///
            // using MfmaFragCMap1d = GetDataLayout_t<MfmaFragC>;
            using MfmaFragDMap1d = GetDataLayout_t<MfmaFragQKD>;

            // MfmaFragC fragsC[BLOCKS_X][BLOCKS_Y];
            // globalReadC(fragsC, c + MfmaFragCMap1d::fromMatrixCoord(warpTileCoord, ldc), ldc);

            ///
            /// Clean up tail A * B
            ///
            MfmaFragQ fragsA[BLOCKS_X];
            MfmaFragK fragsB[BLOCKS_Y];

            // Local read mfma frags
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
            mfma(fragsAcc, fragsA, fragsB, fragsAcc);

            // MfmaFragC fragsD[BLOCKS_X][BLOCKS_Y];
            // #pragma unroll
            // for(int i = 0; i < BLOCKS_X; i++)
            // {
            // #pragma unroll
            //     for(int j = 0; j < BLOCKS_Y; j++)
            //     {
            //         for(int k = 0; k < fragsAcc[i][j].num_elements; k++)
            //         {
            //             // Perform computation in ComputeT and cast back to OutputT
            //             fragsD[i][j].x[k] = __int2float_rz(fragsAcc[i][j].x[k]);
            //         }
            //     }
            // }
            ///
            /// D = alpha * accum + beta * C
            ///
            // MfmaFragD fragsD[BLOCKS_X][BLOCKS_Y];
            // uniformFma(fragsD, alpha, fragsAcc, beta, fragsC);
            globalWriteD(outptr + get<0>(warpTileCoord) * stride_seq_t + get<1>(warpTileCoord), fragsAcc, ldd);
            // outptr += CTA_K;//
        }
    }
}

// ROCWMMA_HOST void simple_ge1d(uint32_t m, uint32_t n, uint32_t k)
// {
//     // Runtime checks for host parameters
//     uint32_t hTBLOCK_X    = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
//     uint32_t hTBLOCK_Y    = isGfx9() ? gfx9Params::TBLOCK_Y : gfx11Params::TBLOCK_Y;
//     uint32_t hBLOCKS_X    = isGfx9() ? gfx9Params::BLOCKS_X : gfx11Params::BLOCKS_X;
//     uint32_t hBLOCKS_Y    = isGfx9() ? gfx9Params::BLOCKS_Y : gfx11Params::BLOCKS_Y;
//     uint32_t hROCWMMA_M   = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
//     uint32_t hROCWMMA_N   = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
//     uint32_t hROCWMMA_K   = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
//     uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
//     uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

//     // Runtime warp calculation (host code needs to query warpsize dynamically)
//     auto warpSize = getWarpSize();
//     auto macroTileSize
//         = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

//     // Device check for supported block and wave sizes
//     if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
//     {
//         std::cout << "Unsupported block size!\n";
//         return;
//     }

//     if(isGfx9() && (hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32))
//     {
//         std::cout << "Unsupported block size!\n";
//         return;
//     }

//     if((isGfx11() || isGfx12()) && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_32)
//     {
//         std::cout << "Unsupported wave size!\n";
//         return;
//     }

//     if(isGfx9() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_64)
//     {
//         std::cout << "Unsupported wave size!\n";
//         return;
//     }

//     // Bounds check
//     if((m < get<0>(macroTileSize) || n < get<1>(macroTileSize) || k < hROCWMMA_K)
//        || (m % hROCWMMA_M || n % hROCWMMA_N || k % hROCWMMA_K))
//     {
//         std::cout << "Unsupported matrix size!\n";
//         return;
//     }

//     // Layouts leading dims
//     int lda = std::is_same_v<DataLayoutQ, row_major> ? k : m;
//     int ldb = std::is_same_v<DataLayoutK, row_major> ? n : k;
//     int ldc = std::is_same_v<DataLayoutS, row_major> ? n : m;
//     int ldd = ldc;

//     std::cout << "Initializing host data..." << std::endl;

//     // Initialize input matrices
//     std::vector<InputT>  matrixA(m * k);
//     std::vector<InputT>  matrixB(k * n);

//     // Fill outputs with NaN to catch contamination
//     std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

//     fillRand(matrixA.data(), m, k);
//     fillRand(matrixB.data(), k, n);

//     std::cout << "Initializing device data..." << std::endl;

//     // Allocate and copy device memory
//     InputT*  d_a;
//     InputT*  d_b;
//     OutputT* d_d;

//     const size_t bytesA = matrixA.size() * sizeof(InputT);
//     const size_t bytesB = matrixB.size() * sizeof(InputT);
//     const size_t bytesD = matrixD.size() * sizeof(OutputT);

//     CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
//     CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
//     CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

//     CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
//     CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
//     CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

//     auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
//     auto tilesM  = rocwmma::ceilDiv(m, get<0>(macroTileSize));
//     auto tilesN  = rocwmma::ceilDiv(n, get<1>(macroTileSize));
//     auto gridDim  = dim3(tilesM * tilesN, 1, 1);

//     std::cout << "Launching GEMM kernel..." << std::endl;
//     std::cout << "gridDim (" << gridDim.x << " " << gridDim.y << " " << gridDim.z << ")"
//               << " blockdim (" << blockDim.x << " " << blockDim.y << ")" << std::endl;

//     // Uses 2 lds blocks for prefetch loop (A and B)
//     int ldsusage
//         = 2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * hROCWMMA_K;

//     ////
//     auto rocwmmaKernel = [&]() {
//         hipExtLaunchKernelGGL(qk_gemm,
//                               gridDim,
//                               blockDim,
//                               ldsusage,
//                               0,
//                               nullptr,
//                               nullptr,
//                               0,
//                               m,
//                               n,
//                               k,
//                               d_a,
//                               d_b,
//                               d_d,
//                               lda,
//                               ldb,
//                               ldd);
//     };

//     constexpr uint32_t warmups    = 2u;
//     constexpr uint32_t recordRuns = 5u;

//     // Warm-up runs, not recorded
//     for(uint32_t i = 0; i < warmups; ++i)
//     {
//         rocwmmaKernel();
//     }

//     // Actual recorded runs
//     hipEvent_t startEvent, stopEvent;
//     CHECK_HIP_ERROR(hipEventCreate(&startEvent));
//     CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

//     CHECK_HIP_ERROR(hipEventRecord(startEvent));
//     for(uint32_t i = 0; i < recordRuns; ++i)
//     {
//         rocwmmaKernel();
//     }
//     CHECK_HIP_ERROR(hipEventRecord(stopEvent));
//     CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

//     auto elapsedTimeMs = 0.0f;
//     CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

//     auto gFlops = calculateGFlops(m, n, k);
//     auto tFlopsPerSec
//         = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs), recordRuns);

//     CHECK_HIP_ERROR(hipEventDestroy(startEvent));
//     CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

//     // Echo performance
//     std::cout << "TBlockX, TBlockY, "
//               << "BlocksX, BlocksY, "
//               << "BlkM, BlkN, BlkK, "
//               << "MatM, MatN, MatK, "
//               << "lda, ldb, ldd,"
//               << "elapsedus, Problem Size(GFlops), TFlops/s" << "," << "sizeof(intputT)" << std::endl;

//     std::cout << hTBLOCK_X << ", " << hTBLOCK_Y << ", " << hBLOCKS_X << ", " << hBLOCKS_Y << ", "
//               << hROCWMMA_M << ", " << hROCWMMA_N << ", " << hROCWMMA_K << ", " << m << ", " << n
//               << ", " << k  << ", " << lda << ", " << ldb << ", " << ldd << ", " << elapsedTimeMs * 200<< ", " << gFlops << ", "
//               << tFlopsPerSec << "," << sizeof(InputT) << std::endl;

// #if !NDEBUG

//     std::cout << "Validating result with reference..." << std::endl;

//     if((uint64_t)m * (uint64_t)n * (uint64_t)k > (2048ull * 2048ull * 2048ull))
//     {
//         std::cout << "Please wait. Large sizes can take a while!" << std::endl;
//     }

//     // Bring kernel result back to host
//     CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

//     // Setup and run reference computation
//     std::vector<OutputT> matrixD_ref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
//     gemm_cpu_h<InputT, OutputT, ComputeT, DataLayoutQ, DataLayoutK, DataLayoutS>(m,
//                                                                                  n,
//                                                                                  k,
//                                                                                  matrixA.data(),
//                                                                                  matrixB.data(),
//                                                                                  matrixD_ref.data(),
//                                                                                  lda,
//                                                                                  ldb,
//                                                                                  ldd);
//     std::cout<<"calculation finished"<<std::endl;
//     auto res = compareEqual(matrixD.data(), matrixD_ref.data(), m * n);

//     if(std::get<0>(res) == false)
//     {
//         std::cout << "FAILED\n";
//     }
//     else
//     {
//         std::cout << "PASSED\n";
//     }

//     std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

// #endif // !NDEBUG

//     // Release device memory
//     CHECK_HIP_ERROR(hipFree(d_a));
//     CHECK_HIP_ERROR(hipFree(d_b));
//     CHECK_HIP_ERROR(hipFree(d_d));

//     std::cout << "Finished!" << std::endl;
// }

// int main()
// {
//     simple_ge1d(4096, 4096, 4096);
//     return 0;
// }