#ifndef _BITONIC_SORT_CUH_
#define _BITONIC_SORT_CUH_

namespace cudamop {

namespace sort {

namespace bitonic {

constexpr int kMaxBitonicSortSize = 4096;

template <typename K, typename V>
struct LTComp {
  __device__ inline bool
  operator()(const K& kA, const V& vA, const K& kB, const V& vB) const {
    return (kA < kB) || ((kA == kB) && (vA < vB));
  }
};

template <typename K, typename V>
struct GTComp {
  __device__ inline bool
  operator()(const K& kA, const V& vA, const K& kB, const V& vB) const {
    return (kA > kB) || ((kA == kB) && (vA < vB));
  }
};

/*!
 * \brief   swap two data
 * \tparam  T               data type
 * \param   t1              data 1
 * \param   t2              data 2
 */
template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

/*!
 * \brief   swap wrapper
 * \tparam  Comparator      comparator for conduct swapping, use LTComp for descending order,
 *                          GTComp for ascending order
 * \tparam  K               type of key
 * \tparam  V               type of value
 * \param   kA              target key A
 * \param   vA              target value A
 * \param   kB              target key B
 * \param   vB              target value B
 * \param   dir             whether need to reverse comparation
 * \param   comp            comparator instance
 */
template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& kA, V& vA,
                                   K& kB, V& vB,
                                   bool dir, const Comparator& comp) {
  bool swap = comp(kA, vA, kB, vB);
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
  }
};

/*!
 * \brief   block-wise bitonic sort
 * \tparam  Comparator      comparator for conduct swapping, use LTComp for ascending order,
 *                          GTComp for descending order
 * \tparam  K               type of key
 * \tparam  V               type of value
 * \tparam  Power2SortSize  #elements to be sorted (should be power of 2)
 * \tparam  ThreadsPerBlock #threads within the block (should be power of 2)
 * \param   keys    keys to be sorted
 * \param   values  values to be sorted (correspond to keys)
 * \param   comp    comparator instance
 */
template<typename Comparator, typename K, typename V, int Power2SortSize, int ThreadsPerBlock>
__device__ inline void bitonicSortBlock(K* keys, V* values, const Comparator& comp) {
    int monotonicLen, stride, loop;

    static_assert(Power2SortSize <= kMaxBitonicSortSize, "sort size <= 4096 only supported");
    static_assert(cudamop::utils::IntegerIsPowerOf2(Power2SortSize), "sort size must be power of 2");
    static_assert(cudamop::utils::IntegerIsPowerOf2(ThreadsPerBlock), "threads in block must be power of 2");

    // e.g., we have 16 elements to be sorted, only 8 comparators are needed
    // #threads == #comparators
    constexpr int numThreadsForSort = Power2SortSize / 2;
    constexpr bool allThreads = numThreadsForSort >= ThreadsPerBlock;

    // if what we are sorting is too large, then threads must loop more than once
    constexpr int loopPerThread = allThreads ? numThreadsForSort / ThreadsPerBlock : 1;

    /*! 
     * \brief   bitonic merge
     * \note    reorder the origin array into a bitonic sequence
     */
#pragma unroll
    for (monotonicLen=2; monotonicLen<Power2SortSize; monotonicLen*=2) {

#pragma unroll
        for (stride=monotonicLen/2; stride>0; stride/=2) {

#pragma unroll
            for (loop=0; loop<loopPerThread; ++loop) {
                // index of comparator
                int comparatorIndex = loop * ThreadsPerBlock + threadIdx.x;

                // judge whether we need to reverse the swap direction as we are 
                // constructing bitonic sequence, e.g.:
                // 1.   assument currently monotonicLen is 4, then we will have 4/2=2 comparators 
                //      within each monotonic squence;
                // 2.   assument currently stride is 2, for the two comparators that charges in element [0-2]
                //      and [1-3] in the 1st monotonic squence, we should reverse the comparation direction;
                // 3.   for the two comparators that charges in element [4-6] and [5-7] in the 2nd monotonic
                //      squence, we should keep the comparation direction;
                // 4.   and so on ...
                bool reverseDirection = ((comparatorIndex & (monotonicLen / 2)) != 0);
                
                // calculate the index of former data to be compared and swapped based on comparator index
                int formerIndex = 2 * comparatorIndex - (comparatorIndex & (stride - 1));

                if (allThreads || (comparatorIndex < numThreadsForSort)) {
                    bitonicSwap<Comparator, K, V>(
                        keys[formerIndex], values[formerIndex],
                        keys[formerIndex + stride], values[formerIndex + stride],
                        reverseDirection, comp);
                }

                __syncthreads();
            }
        }
    }

    /*! 
     * \brief   bitonic sort
     * \note    sorting based on the constructed bitonic sequence
     */
#pragma unroll
    for (stride=Power2SortSize/2; stride>0; stride /= 2) {

#pragma unroll
        for (loop = 0; loop < loopPerThread; ++loop) {
            // index of comparator
            int comparatorIndex = loop * ThreadsPerBlock + threadIdx.x;

            // calculate the index of former data to be compared and swapped based on comparator index
            int formerIndex = 2 * comparatorIndex - (comparatorIndex & (stride - 1));
    
            if (allThreads || (comparatorIndex < numThreadsForSort)) {
                // note: reverse all comparation
                bitonicSwap<Comparator, K, V>(
                  keys[formerIndex], values[formerIndex],
                  keys[formerIndex + stride], values[formerIndex + stride],
                  false, comp
                );
            }

            __syncthreads();
        }
    }
}

} // namespace bitonic

} // namespace sort

} // namespace cudamop

#endif