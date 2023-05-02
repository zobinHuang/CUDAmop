#ifndef _SORT_CUH_
#define _SORT_CUH_

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

template<typename Comparator, typename K, typename V, int Power2SortSize, int ThreadsPerBlock>
__device__ inline void bitonicSortBlock(K* keys, V* values, const Comparator& comp);

#endif