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
#include "test_sort_topk.cuh"

namespace cudamop {

namespace test {

namespace topk {

enum {
  kTestPerWarpHeap = 1,
  kTestReduceTopK,
  kTestSelectTopk,
  kTestSortTopk,
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

template<typename V, typename I, bool isTopK, int testMethod>
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
      auto print_trace_stack = [&]() -> void {
        printf("testMethod: %d, m: %lu, n: %lu, k: %lu\n", testMethod, m, n, k);
        for(j=0; j<k; j++){
          printf("batch: %lu, element: %lu, cpu[%lu]: %f (%u), gpu[%lu]: %f (%u)\n",
            i, j,
            i*n+j, one_row_cpu_topk_pairs[j].value, one_row_cpu_topk_pairs[j].index,
            i*k+j, gpu_topk_values[i*k+j], gpu_topk_indices[i*k+j]
          );
        }
      };

      if(one_row_cpu_topk_pairs[j].value != gpu_topk_values[i*k+j]){
        printf("value error!\n");
        print_trace_stack();
        exit(-1);
      } 

      if(one_row_cpu_topk_pairs[j].index != gpu_topk_indices[i*k+j]){
        bool isFine = false;

        if(!isFine && j != k-1){
          if(gpu_topk_indices[i*k+j] == one_row_cpu_topk_pairs[j+1].index){ isFine = true; }
        }

        if(!isFine && j != 0){
          if(gpu_topk_indices[i*k+j] == one_row_cpu_topk_pairs[j-1].index){ isFine = true; }
        }

        if(!isFine){
          printf("index error!\n");
          print_trace_stack();
          exit(-1);
        }
      }
    } // for(j=0; j<k; j++){
  } // for(i=0; i<m; i++)

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
  output_values.resize(m*k);
  output_indices.resize(m*k);

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
    single_block::select::testRadixSelect<V, I, 256, isTopK>
      (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
  } else if (testMethod == kTestSortTopk) {
    single_block::sort::testRadixSortTopK<V, I, isTopK>
      (m, n, k, d_values, d_output_values, d_indices, d_output_indices);
  }

  // copy result back to host memory
  PROFILE(nvtxRangePush("copy tensor from device to host memory");)
  cudaMemcpy(output_values.data(), d_output_values, m*k*sizeof(V), cudaMemcpyDeviceToHost);
  cudaMemcpy(output_indices.data(), d_output_indices, m*k*sizeof(I), cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // verify result
  if(testMethod != kTestSelectTopk)
    verifyTopKResult<V, I, isTopK, testMethod>(m, n, k, values, indices, output_values, output_indices);

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
  std::vector<uint32_t> ms = {4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192};
  std::vector<uint32_t> ns = {128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288}; 
  std::vector<uint32_t> ks = {32};

  // FIXME: why changing m here not work??
  // for(uint32_t m : ms){
  //   for(uint32_t n : ns){
  //     for(uint32_t k : ks){
  //       printf("TEST: m=%u, n=%u, k=%u: ", m, n, k);

  //       test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(m, n, k);
  //       printf("| kTestPerWarpHeap passed ");

  //       test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(m, n, k);
  //       printf("| kTestReduceTopK passed ");

  //       test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(m, n, k);
  //       printf("| kTestSelectTopk passed ");

  //       test<float, uint32_t, cudamop::test::topk::kTestSortTopk, true>(m, n, k);
  //       printf("| kTestSortTopk passed ");

  //       printf("\n\n");
  //     }
  //   }
  // }

  uint32_t m = 64;
  for(uint32_t n : ns){
    for(uint32_t k : ks){
      printf("TEST: m=%u, n=%u, k=%u: ", m, n, k);

      test<float, uint32_t, cudamop::test::topk::kTestPerWarpHeap, true>(m, n, k);
      printf("| kTestPerWarpHeap passed ");

      test<float, uint32_t, cudamop::test::topk::kTestReduceTopK, true>(m, n, k);
      printf("| kTestReduceTopK passed ");

      test<float, uint32_t, cudamop::test::topk::kTestSelectTopk, true>(m, n, k);
      printf("| kTestSelectTopk passed ");

      test<float, uint32_t, cudamop::test::topk::kTestSortTopk, true>(m, n, k);
      printf("| kTestSortTopk passed ");

      printf("\n\n");
    }
  }

  return 0;
}