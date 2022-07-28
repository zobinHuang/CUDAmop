/*!
 * \file    unified.cu
 * \brief   Unified memory version of vector addition
 * \author  Zhuobin Huang
 * \date    July 27, 2022
 */

#include <iostream>
#include <vector>
#include <nvToolsExt.h>

#include <profile.h>
#include <vector_addition.cuh>

void verifyVectorAdditionResult(
  int *vector_a, 
  int *vector_b, 
  int *vector_c,
  int d
);

int main(){
  // initialize constants
  constexpr int N = 1 << 16;
  constexpr int unit_size = sizeof(int);
  constexpr int vector_size = N*unit_size;

  // data pointers
  int *vector_a, *vector_b, *vector_c;

  // allocate unified memory for vectors
  PROFILE(nvtxRangePush("allocate unified memory for three vectors");)
  cudaMallocManaged(&vector_a, vector_size);
  cudaMallocManaged(&vector_b, vector_size);
  cudaMallocManaged(&vector_c, vector_size);
  PROFILE(PROFILE(nvtxRangePop());)
  
  // initialize source vectors
  PROFILE(nvtxRangePush("initialize source vectors");)
  for(int i=0; i<N; i++){
    vector_a[i] = rand() % 100;
    vector_b[i] = rand() % 100;
  }
  PROFILE(PROFILE(nvtxRangePop());)
  
  // initialize kernel configuration
  // number of kernels per block
  int NUM_THREADS_PER_BLOCK = 1 << 10;

  // number of blocks in the grid
  int NUM_BLOCKS =  N % NUM_THREADS_PER_BLOCK == 0 ?
                    N / NUM_THREADS_PER_BLOCK :
                    N / NUM_THREADS_PER_BLOCK + 1;

  // launch kernel
  PROFILE(
    std::cout << "Launch Kernel: " 
      << NUM_THREADS_PER_BLOCK << " threads per block, " 
      << NUM_BLOCKS << " blocks in the grid" 
      << std::endl;
    nvtxRangePush("start kernel");
  )
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(vector_a, vector_b, vector_c, N);
  
  cudaDeviceSynchronize();
  
  PROFILE(nvtxRangePop();)

  // verify result
  PROFILE(nvtxRangePush("verify result");)
  verifyVectorAdditionResult(vector_a, vector_b, vector_c, N);
  PROFILE(nvtxRangePop();)

  // free device memory
  PROFILE(nvtxRangePush("free unified memory");)
  cudaFree(vector_a);
  cudaFree(vector_b);
  cudaFree(vector_c);
  PROFILE(nvtxRangePop();)

  std::cout << "Get correct vector addition result!" << std::endl;

  return 0;
}

