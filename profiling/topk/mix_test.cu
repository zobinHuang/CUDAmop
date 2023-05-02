/*!
 * \file    per_warp_min_heap.cu
 * \brief   Top-K operation with per-warp min-heap implementation
 * \author  Zhuobin Huang
 * \date    May 3, 2023
 */

#include <iostream>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <random>
#include <nvToolsExt.h>

#include <profile.h>
#include <vector_addition.cuh>

#include "topk.cuh"

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

    // for(i=0; i<m; i++){
    //   for(j=0; j<n; j++){
    //     printf("%f ", values[i*n+j]);
    //   }
    //   printf("\n");
    // }

    // for(i=0; i<m; i++){
    //   for(j=0; j<n; j++){
    //     printf("%u ", indices[i*n+j]);
    //   }
    //   printf("\n");
    // }

    return;
}

template<typename V, typename I>
bool verifyTopKResult(V *values, I *indices, V *gpuTopkValues, I *gpuTopkIndices){

}


template<typename V, typename I>
void test(uint32_t m, uint32_t n, uint32_t k){
  std::vector<V> values;
  values.reserve(m*n);
  std::vector<I> indices;
  indices.reserve(m*n);
  
  // generate random top-k tensors for test
  generateRandomTopKTensors<V, I>(m, n, k, values, indices);

  // TODO:
}

int main(){
  test<float, uint32_t>(1, 1024, 512);
  return 0;
}