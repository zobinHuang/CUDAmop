#ifndef _TEST_SELECT_TOPK_CUH_
#define _TEST_SELECT_TOPK_CUH_

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

#include <topk/sb_radix_select_topk.cuh>

namespace cudamop {

namespace test {

namespace topk {

namespace single_block {

namespace select {

template<typename V, typename I, bool withKthValues, bool isTopK>
constexpr auto radixSelectTopK
    = cudamop::topk::single_block::select::radixSelectTopK<V,I,withKthValues,isTopK>;

template<typename V, typename I, int kBlockSize, bool isTopK>
void testRadixSelect(uint32_t m, uint32_t n, uint32_t k,
    V *d_values, V *d_output_values,
    I *d_indices, I *d_output_indices
){
    PROFILE(
        // std::cout << "Launch Kernel: " 
        // << kBlockSize << " threads per block, " 
        // << m << " blocks in the grid" 
        // << std::endl;
        nvtxRangePush("start kernel");
    )
    radixSelectTopK<V, I, false, true><<<m, kBlockSize>>>
        (d_values, d_output_values, d_output_indices, nullptr, m, n, k);
    PROFILE(
        // cudaDeviceSynchronize is not necessary as following cudaMemcpy would
        // act as synchronization barrier, used here for profiling log
        cudaDeviceSynchronize();
        nvtxRangePop();
    )
}

}

} // namespace single_block

} // namespace topk

} // namespace test

} // namespace cudamop

#endif