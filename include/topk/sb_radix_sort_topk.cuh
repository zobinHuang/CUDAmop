#ifndef _SB_RADIX_SORT_TOPK_CUH_
#define _SB_RADIX_SORT_TOPK_CUH_

#include <cudamop.cuh>
#include <utils/math.cuh>

#include <cub/cub.cuh>


namespace cudamop {

namespace topk {

namespace single_block {

namespace sort {

template<typename V, typename I>
__global__ void copyTempResultToOutput(V *d_temp_values, V *d_values, I *d_temp_indices, I *d_indices, size_t m, size_t n, size_t k){
    size_t j;
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;

    for(j=tid; j<k; j+=blockDim.x){
        d_values[bid*k+j] = d_temp_values[bid*n+j];
        d_indices[bid*k+j] = d_temp_indices[bid*n+j];
    }
}

__global__ void initializerRadixSortTopK(int *begin_offset_buf, int *end_offset_buf, size_t m, size_t n) {
    int i;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if(bid == 0){
        for(i=tid; i<m+1; i+=blockDim.x){
            end_offset_buf[i] = i*n;
            begin_offset_buf[i] = end_offset_buf[i];
        }
    }
}

template<typename V, typename I, bool isTopK>
void radixSortTopK(
    V* input_values,                    // m x n
    I* input_indices,                   // m x n
    V* temp_out_values,                 // m x n
    I* temp_out_indices,                // m x n
    size_t m,                           // #batch
    size_t n,                           // batch size
    size_t k,                           // k
    void *cub_temp_storage,
    size_t &cub_temp_storage_size,
    int* begin_offsets_buf,
    int* end_offsets_buf
){
    // first invoking to obtain the buffer size use by 
    // cub::DeviceSegmentedRadixSort::SortPairsDescending
    if(cub_temp_storage == nullptr){
        isTopK ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
            /* d_temp_storage */ nullptr,
            /* temp_storage_bytes */ cub_temp_storage_size,
            /* d_keys_in */ input_values,
            /* d_keys_out */ (V*) nullptr,
            /* d_values_in */ input_indices,
            /* d_values_out */ (I*) nullptr,
            /* num_items */ m * n,
            /* num_segments */ m,
            /* d_begin_offsets */ begin_offsets_buf,
            /* d_end_offsets */ end_offsets_buf,
            /* begin_bit */ 0,
            /* end_bit */ sizeof(V) * 8
        ) : cub::DeviceSegmentedRadixSort::SortPairs(
            /* d_temp_storage */ nullptr,
            /* temp_storage_bytes */ cub_temp_storage_size,
            /* d_keys_in */ input_values,
            /* d_keys_out */ (V*) nullptr,
            /* d_values_in */ input_indices,
            /* d_values_out */ (I*) nullptr,
            /* num_items */ m * n,
            /* num_segments */ m,
            /* d_begin_offsets */ begin_offsets_buf,
            /* d_end_offsets */ end_offsets_buf,
            /* begin_bit */ 0,
            /* end_bit */ sizeof(V) * 8
        );

        cub_temp_storage_size 
            = cudamop::utils::roundUp<size_t>(cub_temp_storage_size, static_cast<size_t>(256));
        return;
    }

    isTopK ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ cub_temp_storage,
        /* temp_storage_bytes */ cub_temp_storage_size,
        /* d_keys_in */ input_values,
        /* d_keys_out */ temp_out_values,
        /* d_values_in */ input_indices,
        /* d_values_out */ temp_out_indices,
        /* num_items */ m * n,
        /* num_segments */ m,
        /* d_begin_offsets */ begin_offsets_buf,
        /* d_end_offsets */ end_offsets_buf,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(V) * 8
    ) : cub::DeviceSegmentedRadixSort::SortPairs(
        /* d_temp_storage */ cub_temp_storage,
        /* temp_storage_bytes */ cub_temp_storage_size,
        /* d_keys_in */ input_values,
        /* d_keys_out */ temp_out_values,
        /* d_values_in */ input_indices,
        /* d_values_out */ temp_out_indices,
        /* num_items */ m * n,
        /* num_segments */ m,
        /* d_begin_offsets */ begin_offsets_buf,
        /* d_end_offsets */ end_offsets_buf,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(V) * 8
    );

    return;
}

} // namespace sort

} // namespace single_block

} // namespace topk

} // namespace cudamop

#endif