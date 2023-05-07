#ifndef _TEST_SORT_TOPK_CUH_
#define _TEST_SORT_TOPK_CUH_

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <cassert>
#include <random>
#include <nvToolsExt.h>

#include <profile.h>
#include <cudamop.cuh>

#include <topk/sb_radix_sort_topk.cuh>

namespace cudamop {

namespace test {

namespace topk {

namespace single_block {

namespace sort {

template<typename V, typename I, bool isTopK>
constexpr auto radixSortTopK 
    = cudamop::topk::single_block::sort::radixSortTopK<V,I,isTopK>;

template<typename V, typename I>
constexpr auto copyTempResultToOutput
    = cudamop::topk::single_block::sort::copyTempResultToOutput<V,I>;

constexpr auto initializerRadixSortTopK
    = cudamop::topk::single_block::sort::initializerRadixSortTopK;


template<typename V, typename I, bool isTopK>
void testRadixSortTopK(uint32_t m, uint32_t n, uint32_t k,
    V *d_values, V *d_output_values,
    I *d_indices, I *d_output_indices
){
    size_t cub_temp_storage_size;
    void *d_cub_temp_storage = nullptr;

    // allocate temp buffer to store sorting result
    PROFILE(nvtxRangePush("allocate temp buffers");)
    V *d_temp_output_values;
    I *d_temp_output_indices;
    cudaMalloc(&d_temp_output_values, m*n*sizeof(V));
    cudaMalloc(&d_temp_output_indices, m*n*sizeof(I));
    PROFILE(nvtxRangePop();)

    // allocate offset buffer for sorting
    PROFILE(nvtxRangePush("allocate offset buffers");)
    int *d_begin_offset_buf;
    int *d_end_offset_buf;
    cudaMalloc(&d_begin_offset_buf, (m+1)*sizeof(int));
    cudaMalloc(&d_end_offset_buf, (m+1)*sizeof(int));
    PROFILE(nvtxRangePop();)

    // initialize offset buffer
    PROFILE(
        // std::cout << "Launch Init Kernel: " 
        // << 512 << " threads per block, " 
        // << 32 << " blocks in the grid" 
        // << std::endl;
        nvtxRangePush("start init kernel");
    )
    initializerRadixSortTopK<<<32,512>>>(d_begin_offset_buf, d_end_offset_buf, m, n);
    cudaDeviceSynchronize();
    PROFILE(
        nvtxRangePop();
    )
    
    // launch probe kernel to obtain temp buffer size used by cub sorting
    PROFILE(
        // std::cout << "Launch Cub Probe Kernel: " 
        // << std::endl;
        nvtxRangePush("start cub probe kernel");
    )
    radixSortTopK<V, I, isTopK>(
        /* input_values */ d_values,
        /* input_indices */ d_indices, 
        /* temp_out_values */ d_temp_output_values,
        /* temp_out_indices */ d_temp_output_indices,
        /* m */ m,
        /* n */ n,
        /* k */ k,
        /* cub_temp_storage */ d_cub_temp_storage,
        /* cub_temp_storage_size */ cub_temp_storage_size,
        /* begin_offsets_buf */ d_begin_offset_buf,
        /* end_offsets_buf */ d_end_offset_buf+1
    );
    PROFILE(
        cudaDeviceSynchronize();
        nvtxRangePop();
    )

    // allocate temp buffer used by cub sorting
    PROFILE(nvtxRangePush("allocate cub temp buffers");)
    cudaMalloc(&d_cub_temp_storage, cub_temp_storage_size);
    PROFILE(nvtxRangePop();)

    // launch final sort kernel
    PROFILE(
        // std::cout << "Launch Final Sort Kernel: " 
        // << std::endl;
        nvtxRangePush("start final sort kernel");
    )
    radixSortTopK<V, I, isTopK>(
        /* input_values */ d_values,
        /* input_indices */ d_indices, 
        /* temp_out_values */ d_temp_output_values,
        /* temp_out_indices */ d_temp_output_indices,
        /* m */ m,
        /* n */ n,
        /* k */ k,
        /* cub_temp_storage */ d_cub_temp_storage,
        /* cub_temp_storage_size */ cub_temp_storage_size,
        /* begin_offsets_buf */ d_begin_offset_buf,
        /* end_offsets_buf */ d_end_offset_buf+1
    );
    cudaDeviceSynchronize();
    PROFILE(
        nvtxRangePop();
    )

    // copy sorting result to final output buffer
    PROFILE(
        // std::cout << "Launch Copy Result Kernel: " 
        // << 256 << " threads per block, " 
        // << m << " blocks in the grid" 
        // << std::endl;
        nvtxRangePush("start copy result kernel");
    )
    copyTempResultToOutput<V, I><<<m,256>>>
        (d_temp_output_values, d_output_values, d_temp_output_indices, d_output_indices, m, n, k);
    cudaDeviceSynchronize();
    PROFILE(
        nvtxRangePop();
    )
}

} // namespace sort

} // namespace single_block

} // namespace topk

} // namespace test

} // namespace cudamop


#endif