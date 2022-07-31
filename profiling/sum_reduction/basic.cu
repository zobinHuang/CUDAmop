/*!
 * \file    basic.cu
 * \brief   Basic version of sum reduction
 * \author  Zhuobin Huang
 * \date    July 31, 2022
 */

#include <iostream>
#include <vector>
#include <nvToolsExt.h>

#include <profile.h>
#include <sum_reduction.cuh>

void verifySumReductionResult(
  std::vector<int> &source_array,
  std::vector<int> &destination_array
);

int main(){
  // initialize constants
  constexpr int N = 1 << 16;
  constexpr int unit_size = sizeof(int);

  // initialize kernel configuration
  // number of kernels per block
  int NUM_THREADS_PER_BLOCK = 1 << 8;

  // number of blocks in the grid
  int NUM_BLOCKS =  N % NUM_THREADS_PER_BLOCK == 0 ?
                    N / NUM_THREADS_PER_BLOCK :
                    N / NUM_THREADS_PER_BLOCK + 1;

  // obtain size of source and destination array
  constexpr int source_array_size = N*unit_size;
  int dest_array_size = NUM_THREADS_PER_BLOCK*unit_size;

  // obtain share memory size
  int shared_memory_size = NUM_THREADS_PER_BLOCK*unit_size;

  // allocate host memory for data
  PROFILE(nvtxRangePush("allocate host memory for source and destination arrays");)
  std::vector<int> source_array;
  source_array.reserve(N);
  std::vector<int> destination_array;
  destination_array.reserve(NUM_THREADS_PER_BLOCK);
  PROFILE(nvtxRangePop();)

  // generate random number
  PROFILE(nvtxRangePush("initialize source array with random numbers");)
  for(int i=0; i<N; i++){
    source_array.push_back(rand()%100);
  }
  PROFILE(nvtxRangePop();)

  // allocate memory
  PROFILE(nvtxRangePush("allocate device memory for source and destination arrays");)
  int *d_source_array, *d_destination_array;
  cudaMalloc(&d_source_array, source_array_size);
  cudaMalloc(&d_destination_array, dest_array_size);
  PROFILE(nvtxRangePop();)

  // copy source array data
  PROFILE(nvtxRangePush("copy source array from host to device memory");)
  cudaMemcpy(d_source_array, source_array.data(), source_array_size, cudaMemcpyHostToDevice);
  PROFILE(nvtxRangePop();)

  // start first kernel
  PROFILE(
    std::cout << "Launch First Kernel: " 
      << NUM_THREADS_PER_BLOCK << " threads per block, " 
      << NUM_BLOCKS << " blocks in the grid" 
      << std::endl;
    nvtxRangePush("launch first kernel");
  )
  sumReduction<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, shared_memory_size>>>(d_source_array, d_destination_array);
  PROFILE(
    cudaDeviceSynchronize();
    nvtxRangePop();
  )

  // start second kernel
  PROFILE(
    std::cout << "Launch Second Kernel: " 
      << NUM_THREADS_PER_BLOCK << " threads per block, " 
      << 1 << " blocks in the grid" 
      << std::endl;
    nvtxRangePush("launch second kernel");
  )
  sumReduction<<<1, NUM_THREADS_PER_BLOCK>>>(d_destination_array, d_destination_array);
  PROFILE(
    cudaDeviceSynchronize();
    nvtxRangePop();
  )

  // copy result back to host
  PROFILE(nvtxRangePush("copy destination array from device to host memory");)
  cudaMemcpy(destination_array.data(), d_destination_array, dest_array_size, cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // verify result
  PROFILE(nvtxRangePush("verify result");)
  verifySumReductionResult(source_array, destination_array);
  PROFILE(nvtxRangePop();)

  // free device memory
  PROFILE(nvtxRangePush("free device memory");)
  cudaFree(d_source_array);
  cudaFree(d_destination_array);
  PROFILE(nvtxRangePop();)

  std::cout << "Get correct sum reduction result!" << std::endl;

  return 0;
}