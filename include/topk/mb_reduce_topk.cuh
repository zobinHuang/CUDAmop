#ifndef _MB_REDUCE_TOPK_CUH_
#define _MB_REDUCE_TOPK_CUH_

#include <cudamop.cuh>
#include <utils/gpu.cuh>
#include <utils/math.cuh>
#include <cub/cub.cuh>

namespace cudamop {

namespace topk {

namespace multi_block {

namespace reduce {

const float HALF_FLT_MAX = 65504.F;

template<typename V, typename I, bool isTopK>
struct TopKReduceUnit {
    I index = 0;
    V value = -((std::is_same<V, half>::value) ? HALF_FLT_MAX : FLT_MAX);

    __device__ __forceinline__ void insert(V elem, I elem_id){
        if (elem > value) {
            value = elem;
            index = elem_id;
        }
    }

    __device__ __forceinline__ void init(){    
        value = static_cast<V>(-((std::is_same<V, half>::value) ? HALF_FLT_MAX : FLT_MAX));
        index = 0;
    }
};

template<typename V, typename I, bool isTopK>
__device__ __forceinline__ TopKReduceUnit<V, I, isTopK> reduce_topk_op(
        const TopKReduceUnit<V,I,isTopK>& a, const TopKReduceUnit<V,I,isTopK>& b){
    if(isTopK){
        return a.value > b.value ? a : b;
    } else {
        return a.value < b.value ? a : b;
    }
}

template<typename V, typename I, int kBlockSize, int kBlocksPerBatch, bool isTopK>
__global__ void reduceTopKStage1(
    const V* input,             // m x n
    V* temp_input,              // m x n
    V* out_values,              // m x k
    V* temp_out_values,         // m x kBlocksPerBatch x k
    I* out_indices,             // m x k
    I* temp_out_indices,        // m x kBlocksPerBatch x k
    int m,                      // #batch
    int n,                      // batch size
    int k                       // k
) {
    typedef cub::BlockReduce<TopKReduceUnit<V,I,isTopK>, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t elem_id, ite;
    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t batch_id = bid / kBlocksPerBatch;
    const size_t block_lane = bid % kBlocksPerBatch;  

    // where the current batch starts
    const size_t batch_start_index = batch_id * n;
    
    // where the current block write out to temp_out_values and temp_out_indices
    const size_t temp_out_start_index = batch_id * kBlocksPerBatch * k + block_lane * k;

    TopKReduceUnit<V, I, isTopK> partial;
    const bool IS_FP16   = std::is_same<V, half>::value;
    const V    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // copy the origin values to the temp buffer
    for(elem_id=tid+block_lane*kBlockSize; elem_id<n; elem_id+=kBlockSize*kBlocksPerBatch){
        size_t index = elem_id + batch_start_index;
        temp_input[index] = input[index];
    }

    // reduce for k times to find out top-ks within current data block
    for(ite=0; ite<k; ite++) {
        partial.init();
        
        // thread-wise reduce
#pragma unroll
        for(elem_id=tid+block_lane*kBlockSize; elem_id<n; elem_id+=kBlockSize*kBlocksPerBatch){
            size_t index = elem_id + batch_start_index;
            partial.insert(temp_input[index], static_cast<I>(index));
        }

        // block-wise reduce
        TopKReduceUnit<V, I, isTopK> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<V,I,isTopK>);

        // write out the result, and change the input buffer
        if(tid == 0){
            const int index = temp_out_start_index + ite;
            temp_out_values[index] = total.value;
            temp_out_indices[index] = total.index;
            temp_input[total.index] = -MAX_T_VAL;
        }

        __syncthreads();
    }
}

template<typename V, typename I, int kBlockSize, int kBlocksPerBatchStage1, bool isTopK>
__global__ void reduceTopKStage2(
    const V* input,             // m x n
    V* temp_input,              // m x n
    V* out_values,              // m x k
    V* temp_out_values,         // m x kBlocksPerBatchStage1 x k
    I* out_indices,             // m x k
    I* temp_out_indices,        // m x kBlocksPerBatchStage1 x k
    int m,                      // #batch
    int n,                      // batch size
    int k                       // k
) {
    typedef cub::BlockReduce<TopKReduceUnit<V, I, isTopK>, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage         temp_storage;

    const bool IS_FP16   = std::is_same<V, half>::value;
    const V    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    const int tid      = threadIdx.x;
    const int batch_id = blockIdx.x;

    size_t elem_id, ite;

    // where the current block read from temp_out_values and temp_out_indices
    const size_t batch_temp_start_index = batch_id * kBlocksPerBatchStage1 * k;

    // where the current block write to out_values and out_indices
    const size_t batch_out_index = batch_id * k;

    TopKReduceUnit<V, I, isTopK> partial;

    for(ite=0; ite<k; ite++){
        partial.init();

        // thread-wise reduce
#pragma unroll
        for(elem_id=tid; elem_id<kBlocksPerBatchStage1*k; elem_id+=kBlockSize){
            const size_t index = batch_temp_start_index + elem_id;
            partial.insert(temp_out_values[index], index);
        }

        // block-wise reduce
        TopKReduceUnit<V, I, isTopK> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<V,I,isTopK>);

        // write out the result, and change the input buffer
        if(tid == 0){
            const int index = batch_out_index + ite;
            out_values[index] = total.value;
            out_indices[index] = temp_out_indices[total.index] % n;
            temp_out_values[total.index] = -MAX_T_VAL;
        }

        __syncthreads();
    }
}

} // namespace reduce

} // namespace multi_block

} // namespace topk

} // namespace cudamop

#endif