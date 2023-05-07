#ifndef _TEST_REDUCE_TOPK_CUH_
#define _TEST_REDUCE_TOPK_CUH_

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <cassert>
#include <random>
#include <nvToolsExt.h>

#include <profile.h>
#include <vector_addition.cuh>
#include <cudamop.cuh>

#include <topk/mb_reduce_topk.cuh>

namespace cudamop {

namespace test {

namespace topk {

namespace multi_block {

namespace reduce {

template<typename V, typename I, int kBlockSize, int kBlocksPerBatch, bool isTopK>
constexpr auto reduceTopKStage1 
    = cudamop::topk::multi_block::reduce::reduceTopKStage1<V,I,kBlockSize,kBlocksPerBatch,isTopK>;

template<typename V, typename I, int kBlockSize, int kBlocksPerBatchStage1, bool isTopK>
constexpr auto reduceTopKStage2 
    = cudamop::topk::multi_block::reduce::reduceTopKStage2<V,I,kBlockSize,kBlocksPerBatchStage1,isTopK>;

template<typename V, typename I, int kBlockSize_1, int kBlockSize_2, int kBlockPerBatch_1, bool isTopK>
void testReduce(uint32_t m, uint32_t n, uint32_t k,
    V *d_values, V *d_output_values,
    I *d_indices, I *d_output_indices
){
  PROFILE(nvtxRangePush("allocate temp buffers");)
  V *d_temp_values;
  V *d_temp_output_values;
  I *d_temp_output_indices;
  cudaMalloc(&d_temp_values, m*n*sizeof(V));
  cudaMalloc(&d_temp_output_values, m*kBlockPerBatch_1*k*sizeof(V));
  cudaMalloc(&d_temp_output_indices, m*kBlockPerBatch_1*k*sizeof(I));
  PROFILE(nvtxRangePop();)

  PROFILE(
    // std::cout << "Launch Stage 1 Kernel: " 
    //   << kBlockSize_1 << " threads per block, " 
    //   << kBlockPerBatch_1 * m << " blocks in the grid"
    //   << std::endl;
    nvtxRangePush("start stage 1 kernel");
  )
  reduceTopKStage1<V, I, kBlockSize_1, kBlockPerBatch_1, isTopK>
    <<<m*kBlockPerBatch_1,kBlockSize_1>>>(d_values, d_temp_values, d_output_values, d_temp_output_values, d_output_indices, d_temp_output_indices, m, n, k);
  cudaDeviceSynchronize();
  PROFILE(
    nvtxRangePop();
  )

  PROFILE(
    // std::cout << "Launch Stage 2 Kernel: " 
    //   << kBlockSize_2 << " threads per block, " 
    //   << m << " blocks in the grid"
    //   << std::endl;
    nvtxRangePush("start stage 2 kernel");
  )
  reduceTopKStage2<V, I, kBlockSize_2, kBlockPerBatch_1, isTopK>
    <<<m,kBlockSize_2>>>(d_values, d_temp_values, d_output_values, d_temp_output_values, d_output_indices, d_temp_output_indices, m, n, k);
  cudaDeviceSynchronize();
  PROFILE(
    nvtxRangePop();
  )
}

} // namespace reduce

} // namespace multi_block

} // namespace topk

} // namespace test

} // namespace cudamop

#endif