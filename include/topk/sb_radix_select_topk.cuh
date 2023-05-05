#ifndef _SB_SELECT_CUH_
#define _SB_SELECT_CUH_

#include <cudamop.cuh>
#include <utils/gpu.cuh>
#include <utils/math.cuh>
#include <utils/scan.cuh>
#include <select/radix_select.cuh>
#include <cub/cub.cuh>

namespace cudamop {

namespace topk {

namespace single_block {

namespace select {

template<typename V, typename I, bool withKthValues, bool isTopK>
__global__ void radixSelectTopK(
    V* input,                   // m x n
    V* out_values,              // m x k
    I* out_indices,             // m x k
    V* kthValues,               // 
    size_t m,                   // #batch
    size_t n,                   // batch size
    size_t k                    // k
) {
    __shared__ unsigned int smem[kWarpSize];
    size_t i;

    // one block per batch, hence block id is batch id
    size_t batch_id = utils::getLinearBlockId<size_t>();
    if(batch_id >= m){
        return;
    }

    // extract the start address of current block
    V* batch_input = &input[batch_id * n];
    V* batch_out_values = &out_values[batch_id * k];
    I* batch_out_indices = &out_indices[batch_id * k];

    // find the top-kth (bottom-kth) element in our input
    V topKValue;
    if(withKthValues){
        topKValue = kthValues[batch_id];
    } else {
        topKValue = static_cast<V>(0);
        cudamop::select::radix::radixSelectTopKth<V, isTopK>(batch_input, k, n, smem, &topKValue);
    }
    const auto topKConverted = cudamop::select::radix::TypeTranslator<V>::convert(topKValue);

    /*!
     * \note    the following code is to find out all values that is greater (less) than
     *          the top-kth (bottom-kth) value
     * \note    since we will do exclusive prefix scan below, we need all threads within
     *          the block to participate in processing, so we round up number of iterations
     *          to the multiple of blockDim.x
     */
    size_t numIterations = cudamop::utils::roundUp<size_t>(n, static_cast<size_t>(blockDim.x));

    size_t writeIndexStart = 0;

    for(i=threadIdx.x; i<numIterations; i+=blockDim.x){
        // extract the value if in range
        bool inRange = (i < n);
        V value = inRange ? batch_input[i] : static_cast<V>(0);
        const auto convertedValue = cudamop::select::radix::TypeTranslator<V>::convert(value);

        // check whether current thread loads a value greater (less) than top-kth value
        bool hasBeforeTopK;
        if(isTopK){
            hasBeforeTopK = inRange && (convertedValue > topKConverted);
        } else {
            hasBeforeTopK = inRange && (convertedValue < topKConverted);
        }

        /*!
         * \note    conduct a prefix binary scan to know how many threads before current thread
         *          load a value greater (less) than top-kth (bottom-kth) value:
         *          [1] numPrevHasBeforeTopK is the above's number;
         *          [2] numAllHasBeforeTopK is the total number of thread that load a value greater 
         *              (less) than top-kth value within current block;
         */
        unsigned int numPrevHasBeforeTopK, numAllHasBeforeTopK;
        cudamop::utils::exclusiveBinaryPrefixScan<unsigned int, true>
            (smem, hasBeforeTopK, &numPrevHasBeforeTopK, &numAllHasBeforeTopK, utils::PrefixScanAddOp<unsigned int>());

        if(hasBeforeTopK){
            size_t writeIndex = writeIndexStart + numPrevHasBeforeTopK;
            assert(writeIndex < k);
            batch_out_values[writeIndex] = value;
            batch_out_indices[writeIndex] = i;
        }
        
        writeIndexStart += numAllHasBeforeTopK;
    }
    
    // calculate how many top-kth (bottom-kth) value remained to be written out
    assert(k >= writeIndexStart);
    size_t topKRemaining = (k - writeIndexStart);

    // find out the remaining top-kth (bottom-kth) values
    for(i=threadIdx.x; i<numIterations; i+=blockDim.x){
        // extract the value if in range
        bool inRange = (i<n);
        V value = inRange ? batch_input[i] : static_cast<V>(0);
        const auto convertedValue = cudamop::select::radix::TypeTranslator<V>::convert(value);

        // check whether current thread loads a value equal to the top-kth value
        bool hasTopK;
        hasTopK = inRange && (convertedValue == topKConverted);

        // exclusive binary prefix sum
        unsigned int numPrevHasTopK, numAllHasTopK;
        cudamop::utils::exclusiveBinaryPrefixScan<unsigned int, true>
            (smem, hasTopK, &numPrevHasTopK, &numAllHasTopK, cudamop::utils::PrefixScanAddOp<unsigned int>());

        // we only write out the values when the number of all previous threads
        // that hold top-kth value is still less than the remaining top-kth value
        // to find out, and of course when current thread has the top-kth value
        if(hasTopK && (numPrevHasTopK < topKRemaining)) {
            size_t writeIndex = writeIndexStart + numPrevHasTopK;
            assert(writeIndex < k);
            batch_out_values[writeIndex] = value;
            batch_out_indices[writeIndex] = i;
        }

        // we leave the loop while we have already fill all k values
        if(numAllHasTopK >= topKRemaining){
            break;
        }

        topKRemaining -= numAllHasTopK;
        writeIndexStart += numAllHasTopK;
    }
}

} // namespace select

} // namespace single_block

} // namespace topk

} // namespace cudamop

#endif