/*!
 * \file    per_warp_min_heap.cu
 * \brief   Top-K operation with per-warp min-heap implementation
 * \author  Zhuobin Huang
 * \date    May 3, 2023
 */

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

#include "test_reduce_topk.cuh"
#include "test_heap_topk.cuh"
#include "test_select_topk.cuh"

namespace cudamop {

namespace test {

namespace topk {

enum {
  kTestPerWarpHeap = 1,
  kTestReduceTopK,
  kTestSelectTopk, 
};

template<typename V, typename I>
void generateRandomTopKTensors(
      uint64_t m, uint64_t n, uint64_t k, 
      std::vector<V> &values, std::vector<I> &indices
){
    uint64_t i, j;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<V> dist(0, 1);

    srand(static_cast<unsigned>(time(0)));

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            V random_p = static_cast<V>(0);

            // random_p = static_cast<V>(rand()) / static_cast<V>(RAND_MAX/p_budget);
            random_p = dist(gen);

            values.push_back(static_cast<V>(random_p));
            indices.push_back(static_cast<I>(j));
        }
    }

    // for(i=0; i<m; i++){ for(j=0; j<n; j++){ printf("%f ", values[i*n+j]); } printf("\n"); }
    // for(i=0; i<m; i++){ for(j=0; j<n; j++){ printf("%u ", indices[i*n+j]); } printf("\n"); }

    return;
}

template<typename V, typename I>
struct topk_pair{ V value; I index; };

template<typename V, typename I>
bool greater_sort(topk_pair<V,I> a, topk_pair<V,I> b){ return (a.value > b.value); }

template<typename V, typename I>
bool less_sort(topk_pair<V,I> a, topk_pair<V,I> b){ return (a.value < b.value); }

template<typename V, typename I, bool isTopK>
bool verifyTopKResult(
  uint64_t m, uint64_t n, uint64_t k,
  std::vector<V> &values, std::vector<I> &indices, 
  std::vector<V> &gpu_topk_values, std::vector<I> &gpu_topk_indices
){
  uint64_t i, j;

  std::vector< std::vector<topk_pair<V,I>> > cpu_topk_pairs;
  cpu_topk_pairs.reserve(m);

  for(i=0; i<m; i++){
    std::vector<topk_pair<V,I>> one_row_cpu_topk_pairs;
    one_row_cpu_topk_pairs.reserve(n);

    for(j=0; j<n; j++){
      topk_pair<V,I> tp;
      tp.index = j;
      tp.value = values[i*n+j];
      one_row_cpu_topk_pairs.push_back(tp);
    }

    if(isTopK){
      std::sort(one_row_cpu_topk_pairs.begin(), one_row_cpu_topk_pairs.end(), greater_sort<V,I>);
    } else {
      std::sort(one_row_cpu_topk_pairs.begin(), one_row_cpu_topk_pairs.end(), less_sort<V,I>);
    }

    cpu_topk_pairs.push_back(one_row_cpu_topk_pairs);
  }

  for(i=0; i<m; i++){
    std::vector<topk_pair<V,I>> &one_row_cpu_topk_pairs = cpu_topk_pairs[i];

    // printf("cpu kth value: %f\n", one_row_cpu_topk_pairs[k-1].value);

    for(j=0; j<k; j++){
      printf("cpu[%lu]: %f (%u), gpu[%lu]: %f (%u)\n",
        i*n+j, one_row_cpu_topk_pairs[j].value, one_row_cpu_topk_pairs[j].index,
        i*k+j, gpu_topk_values[i*k+j], gpu_topk_indices[i*k+j]
      );
      // assert(one_row_cpu_topk_pairs[j].value == gpu_topk_values[i*k+j]);
      // assert(one_row_cpu_topk_pairs[j].index == gpu_topk_indices[i*k+j]);
    }
  }

  return true;
}

template<typename V, typename I, int testMethod, bool isTopK>
void test(uint32_t m, uint32_t n, uint32_t k){
  std::vector<V> values;
  std::vector<I> indices;
  std::vector<V> output_values;
  std::vector<I> output_indices;
  V *d_values;
  I *d_indices;
  V *d_output_values;
  I *d_output_indices;
  
  values.reserve(m*n);
  indices.reserve(m*n);
  output_values.reserve(m*k);
  output_indices.reserve(m*k);

  PROFILE(nvtxRangePush("generate random topk tensors");)
  generateRandomTopKTensors<V, I>(m, n, k, values, indices);
  PROFILE(nvtxRangePop();)

  PROFILE(nvtxRangePush("allocate device memory for tensors");)
  cudaMalloc(&d_values, m*n*sizeof(V));
  cudaMalloc(&d_indices, m*n*sizeof(I));
  cudaMalloc(&d_output_values, m*k*sizeof(V));
  cudaMalloc(&d_output_indices, m*k*sizeof(I));
  PROFILE(nvtxRangePop();)

  PROFILE(nvtxRangePush("copy tensors from host to device memory");)
  cudaMemcpy(d_values, values.data(), m*n*sizeof(V), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices.data(), m*n*sizeof(I), cudaMemcpyHostToDevice);
  PROFILE(nvtxRangePop();)

  if (testMethod == kTestPerWarpHeap) {
    if (k < 32) {
      single_block::heap::testPerWarpHeap<V, I, 256, 32, isTopK>
        (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
    } else if (k < 128) {
      single_block::heap::testPerWarpHeap<V, I, 256, 128, isTopK>
        (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
    } else if (k < 512) {
      single_block::heap::testPerWarpHeap<V, I, 256, 512, isTopK>
        (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
    } else {
      fprintf(stderr, "testPerWarpHeap (single block) can't support k that exceed 512");
      goto release_device_memory;
    }
  } else if (testMethod == kTestReduceTopK) {
      if (k >= 1 && k <= 16) {
        multi_block::reduce::testReduce<V, I, 128, 128, 8, isTopK>
          (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
      } else if (k >= 17 && k <= 32) {
        multi_block::reduce::testReduce<V, I, 256, 128, 8, isTopK>
          (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
      } else if (k >= 33 && k <= 64) {
        multi_block::reduce::testReduce<V, I, 256, 256, 8, isTopK>
          (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
      } else if (k >= 65 && k <= 1024) {
        multi_block::reduce::testReduce<V, I, 256, 256, 8, isTopK>
          (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
      } else {
        fprintf(stderr, "testReduce (multiple block) can't support k that exceed 1024");
        goto release_device_memory;
      }
  } else if (testMethod == kTestSelectTopk) {
    single_block::select::testradixSelect<V, I, 256, isTopK>
      (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
  }

  // copy result back to host memory
  PROFILE(nvtxRangePush("copy tensor from device to host memory");)
  cudaMemcpy(output_values.data(), d_output_values, m*k*sizeof(V), cudaMemcpyDeviceToHost);
  cudaMemcpy(output_indices.data(), d_output_indices, m*k*sizeof(I), cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // verify result
  verifyTopKResult<V, I, isTopK>(m, n, k, values, indices, output_values, output_indices);

release_device_memory:
  PROFILE(nvtxRangePush("free device memory");)
  cudaFree(d_values);
  cudaFree(d_indices);
  cudaFree(d_output_values);
  cudaFree(d_output_indices);
  PROFILE(nvtxRangePop();)

  return;
}

} // namespace topk

} // namespace test

} // namespace cudamop

template<typename V, typename I, int testMethod, bool isTopK>
constexpr auto test = cudamop::test::topk::test<V,I,testMethod,isTopK>;

int main(){
  // test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(1, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(2, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(8, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(16, 512, 128);

  // test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(1, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(2, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(8, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(16, 512, 128);

  test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(2, 64, 32);
  // test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(2, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(8, 512, 128);
  // test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(16, 512, 128);

  return 0;
}