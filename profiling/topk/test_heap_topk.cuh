#ifndef _TEST_HEAP_TOPK_CUH_
#define _TEST_HEAP_TOPK_CUH_

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

#include <topk/sb_perwarp_heap_topk.cuh>

namespace cudamop {

namespace test {

namespace topk {

namespace single_block {

namespace heap {

template <typename V, typename I, int ThreadsPerBlock, int HeapSize, bool isMinHeap>
constexpr auto perWarpHeapTopK
    = cudamop::topk::single_block::heap::perWarpHeapTopK<V,I,ThreadsPerBlock,HeapSize,isMinHeap>;

template<typename V, typename I, int kBlockSize, int kHeapSize, bool isTopK>
void testPerWarpHeap(uint32_t m, uint32_t n, uint32_t k,
    V *d_values, V *d_output_values,
    I *d_indices, I *d_output_indices
){
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int smem = kNumWarps * kHeapSize * (sizeof(V) + sizeof(I));
  constexpr V kInitValue = std::numeric_limits<V>::lowest();
  constexpr I kInitIndex = std::numeric_limits<I>::lowest();

  PROFILE(
    std::cout << "Launch Kernel: " 
      << kBlockSize << " threads per block, " 
      << m << " blocks in the grid" 
      << std::endl;
    nvtxRangePush("start kernel");
  )
  perWarpHeapTopK<V, I, kBlockSize, kHeapSize, isTopK><<<m, kBlockSize, smem>>>
    (d_values, d_output_values, d_output_indices, kInitValue, kInitIndex, m, n, k);
  PROFILE(
    // cudaDeviceSynchronize is not necessary as following cudaMemcpy would
    // act as synchronization barrier, used here for profiling log
    cudaDeviceSynchronize();
    nvtxRangePop();
  )
}

} // namespace heap

} // namespace single_block

} // namespace topk

} // namespace test

} // namespace cudamop

#endif