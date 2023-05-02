#ifndef _TOP_K_CUH_
#define _TOP_K_CUH_

template <typename V, typename I, typename OutIndexType, int ThreadsPerBlock, int HeapSize, bool isMinHeap>
__global__ void perWarpHeapTopK(const V* input, V* outValues, OutIndexType* outIndices, V initVal, I initIndex, 
        int m, int n, int k);

#endif