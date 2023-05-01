/*!
 * \file    min_heap_topk.cu
 * \brief   Operator for sort-based top-k operation
 * \author  Zhuobin Huang
 * \date    May. 1, 2023
 */

#include "cudamop.cuh"
#include "gpu_utils.cuh"
#include "math_util.cuh"

/*!
 * \brief   obtain the overall share memory size that the heap occupies
 *          within the block
 * \tparam  perWarpHeap maintain granularity of the heap, true for per-warp, false for per-thread
 * \param   valueSize   size of the value type
 * \param   indexSize   size of the index type
 * \param   numThreads  number of thread within the block
 * \param   heapSize    size of a single heap
 * \return  the overall share memory size that the heap occupies within the block
 */
template<bool perWarpHeap>
constexpr size_t getHeapSmemSize(size_t valueSize, size_t indexSize, int numThreads, int heapSize) {
    if(perWarpHeap)
        return (numThreads / kWarpSize) * heapSize * (valueSize + indexSize);
    else
        return numThreads * heapSize * (valueSize + indexSize);
}

/*!
 * \brief   definition of the heap structure, with maintain granularity of per-warp
 * \tparam  V                   type of the value
 * \tparam  I                   type of the index
 * \tparam  ThreadsPerBlock     #threads within the block
 * \tparam  HeapSize            size of a single heap
 * \tparam  isMinHeap           true for min-heap, false for max-heap
 */
template <typename V, typename I, int ThreadsPerBlock, int HeapSize, bool isMinHeap>
class PerWarpHeap {
  public:
    /*!
     * \brief   return the shared memory size occupied by heap within the block
     * \return  the overall share memory size that the heap occupies within the block
     */
    static constexpr size_t getSmemSize() {
        return getHeapSmemSize<true>(sizeof(V), sizeof(I), ThreadsPerBlock, HeapSize);
    }

    /*!
     * \brief   constructor function
     * \param   smem        the pointer to the shared memory area
     * \param   initValue   initial content of all values
     * \param   initIndex   initial content of all indices
     */
    __device__ PerWarpHeap(void *smem, V initValue, I initIndex) {
        heapBase = smem;
        int i;
        int warpId = threadIdx.x / kWarpSize;
        int laneId = getLaneId();

        auto vStart = getValueStart();
        heapValues = &vStart[warpId * HeapSize];
        auto iStart = getIndicesStart();
        heapIndices = &iStart[warpId * HeapSize];
        
    #pragma unroll
        for(i=laneId; i<HeapSize; i+=kWarpSize){
            heapValues[i] = initValue;
            heapIndices[i] = initIndex;
        }
    }

    /*!
     * \brief   return a pointer to the start of our block-wide value storage
     * \return  a pointer to the start of our block-wide value storage
     */
    inline __device__ V* getValueStart() {
        return (V*)heapBase;
    }

    /*!
     * \brief   return a pointer to the start of our block-wide indices storage
     * \return  a pointer to the start of our block-wide indices storage
     */
    inline __device__ I* getIndicesStart() {
        constexpr int warpsPerBlock = ThreadsPerBlock / kWarpSize;
        return (I*)&getValueStart()[warpsPerBlock * HeapSize];
    }
    
    /*!
     * \brief   return a pointer past the end of our block-wide heap storage
     * \return  a pointer past the end of our block-wide heap storage
     */
    inline __device__ void* getStorageEnd() {
        constexpr int warpsPerBlock = ThreadsPerBlock / kWarpSize;
        return getIndicesStart() + (warpsPerBlock * HeapSize);
    }

    /*!
     * \brief   attempt to add a new value with corresponding index into the heap
     * \param   value   the value to be added
     * \param   index   the index to be added
     */
    inline __device__ void add(V value, I index) {
        int i;
        bool wantInsert = isMinHeap ? (value > heapHead) : (value < heapHead);

        // find out all the lanes that have elements to add to the heap
        // within the warp
        unsigned int vote = __ballot_sync(__activemask(), wantInsert);

        if (!vote) {
            // everything the warp has is smaller than our heap
            return;
        }
        
        // calculate how many thread before current thread within the warp want insert
        int wantInsertIndex = __popc(getLaneMaskLt() & vote);

        // calculate the overall number of threads that want insert
        int total = __popc(vote);

        /*!
         *  \note   we below serialize execution of heap insertion, under SIMT, the pace
         *          of each thread within the same warp should be the same
         */
        for (i = 0; i < total; ++i) {
            if (wantInsertIndex == i && wantInsert) {
                // insert into the heap (insertion and reordering)
                _warpHeapInsert(value, index);

                /*!
                 *  \note   prevent the compiler optimization of caching the write to shared
                 *          memory in the register, make sure all smem writes are immediately
                 *          visible to other thread within the warp
                 *  \note   is there no warp-wise shared memory fence?   
                 *  \ref    https://stackoverflow.com/a/5243177/14377282
                 */
                __threadfence_block();
            }
        }

        // record the heap head to the local register
        heapHead = heapValues[0];
    }

    /*!
     * \brief   reduce all per-warp heaps to a unified, sorted list
     */
    inline __device__ void reduceHeaps() {
        // TODO:
    }

    /*!
     * \brief   insert a new value with corresponding index into the heap (called by PerWarpHeap::add)
     * \note    this device function is executed one thread by one thread within the warp
     * \param   value   the value to be added
     * \param   index   the index to be added
     */
    __device__ inline void _warpHeapInsert(V value, I index) {
        /*!
         * \note    since previous threads within the warp might insert new value into the heap,
         *          so we double check here again
         */
        bool valid = isMinHeap ? (value > heapValues[0]) : (value < heapValues[0]);
         if (!valid) {
            return;
        }

        V currentValue = value;
        I currentIndex = index;

        // swap with head if valid
        heapValues[0] = value;
        heapIndices[0] = index;

        // heapify (reordering the heap)
        int currentHeapPos = 0;
    #pragma unroll
        /*!
         * \note    we assume our heap is full binary tree, hence The number of interior nodes 
         *          in the heap is log2(HeapSize / 2), for example:
         *          heap size 8 means there are 7 elements in the heap, indices 0-6 (0 12 3456)
         *          log2(8 / 2) = 2 levels of interior nodes for heap size 8 (0 and 12)
         */
        for (int levels = 0; levels < IntegerLog2(HeapSize / 2); ++levels) {
            int leftChildPos = currentHeapPos * 2 + 1;
            int rightChildPos = leftChild + 1;
            V leftChildValue = heapValues[leftChildPos];
            V rightChildValue = heapValues[rightChildPos];

            // choose the child to swap (larger child while using max-heap, smaller child while using min-heap)
            bool swapLeft = isMinHeap ? (leftChildValue < rightChildValue) : (leftChildValue > rightChildValue);
            int childPosToSwap = swapLeft ? leftChildPos : rightChildPos;
            V childValueToSwap = swapLeft ? leftChildValue : rightChildValue;
            
            // judge whether need to do swaping
            valid = isMinHeap ? currentValue >= childValueToSwap : currentValue <= childValueToSwap;

            // do swaping
            heapValues[currentHeapPos] = valid ? childValueToSwap : currentValue;
            heapIndices[currentHeapPos] = valid ? heapIndices[childPosToSwap] : currentIndex;
            heapValues[childPosToSwap] = valid ? currentValue : childValueToSwap;
            heapIndices[childPosToSwap] = valid ? currentIndex : heapIndices[childPosToSwap];

            // update current position, value and corresponding index
            currentHeapPos = childPosToSwap;
            currentValue = heapValues[currentHeapPos];
            currentIndex = heapIndices[currentHeapPos];
        }
    }

  private:
    // start address of all heaps within the shared memory
    void *heapBase;  
    
    // heap head of current warp
    V heapHead;
    
    // start address of the values part within the heap of current warp
    V *heapValues;
    
    // start address of the indices part within the heap of current warp
    I *heapIndices;
};