#ifndef _SCAN_UTILS_CUH_
#define _SCAN_UTILS_CUH_

#include <cudamop.cuh>
#include <utils/gpu.cuh>

namespace cudamop {

namespace utils {

/*!
 * \brief   prefix scan adding binary operator
 */
template <typename T>
struct PrefixScanAddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

/*!
 * \brief   conduct prefix inclusive scan across all threads within the block
 *          based on given binary op
 * \tparam  T                   type of data
 * \tparam  killWARDependency   ?
 * \tparam  BinaryFunction      class of given scan binary function
 * \param   smem    shared memory area for internal usage, size should
 *                  be #Warp * sizeof(T)
 * \param   in      vote of current thread
 * \param   out     final prefix scan result of current thread
 * \param   binOp   given scan binary function
 */
template<typename T, bool killWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(T *smem, bool in, T *out, BinaryFunction binOp){
    size_t i;

    // within-warp voting
    T vote = getWarpBallot(in);

    // how many threads voted positive before and including current thread
    // within the current warp
    T index = __popc(getLaneMaskLe() & vote);

    // how many threads voted positive within the current warp
    T carry = __popc(vote);

    size_t warpId = threadIdx.x / kWarpSize;
    size_t numWarps = blockDim.x / kWarpSize;

    // for each warp, write out the value
    if(getLaneId() == 0){
        smem[warpId] = carry;
    }
    __syncthreads();

    // sum across warps in one thread
    if(threadIdx.x == 0) {
        size_t prev = 0;
        for(i=0; i<numWarps; i++){
            T value = smem[i];
            smem[i] = binOp(smem[i], prev);
            prev = binOp(prev, value);
        }
    }
    __syncthreads();

    // including the vote result before and include current thread within
    // the warp, to the previous warp's scan result, as the final scan result
    // of current thread
    if(warpId >= 1){
        index = binOp(index, smem[warpId-1]);
    }

    *out = index;

    if(killWARDependency){
        __syncthreads();
    }
}

/*!
 * \brief   conduct prefix exclusive scan across all threads within the block
 *          based on given binary op
 * \tparam  T                   type of data
 * \tparam  killWARDependency   ?
 * \tparam  BinaryFunction      class of given scan binary function
 * \param   smem    shared memory area for internal usage, size should
 *                  be #Warp * sizeof(T)
 * \param   in      vote of current thread
 * \param   out     final prefix scan result of current thread
 * \param   carry   the overall prefix scan result of current block
 * \param   binOp   given scan binary function
 */
template<typename T, bool killWARDependency, class BinaryFunction>
__device__ void exclusiveBinaryPrefixScan(T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
    inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

    // inclusive to exclusive
    *out -= (T) in;

    // the outgoing carry for all threads is the last warp's sum
    *carry = smem[ceilDiv<size_t>(blockDim.x, kWarpSize) - 1];

    if (killWARDependency) {
        __syncthreads();
    }
}


} // namespace utils

} // namespace cudamop

#endif