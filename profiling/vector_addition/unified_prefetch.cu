/*!
 * \file    prefetch_unified.cu
 * \brief   Unified memory version of vector addition
 *          with asynchronized prefetching optimization
 * \author  Zhuobin Huang
 * \date    July 28, 2022
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
    int d);

int main(){
  // initialize constants
  constexpr int N = 1 << 16;
  constexpr int unit_size = sizeof(int);
  constexpr int vector_size = N*unit_size;

  // get cuda device index
  int gpu_id = cudaGetDevice(&gpu_id);

  // data pointers
  int *vector_a, *vector_b, *vector_c;

  // allocate unified memory for vectors
  PROFILE(nvtxRangePush("allocate unified memory for three vectors");)
  cudaMallocManaged(&vector_a, vector_size);
  cudaMallocManaged(&vector_b, vector_size);
  cudaMallocManaged(&vector_c, vector_size);
  PROFILE(PROFILE(nvtxRangePop());)
  
  // put vector_a and vector_b at cpu side for initialization
  // put vector_c at gpu side
  cudaMemAdvise(vector_a, vector_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(vector_b, vector_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(vector_c, vector_size, gpu_id);

  // initialize source vectors
  PROFILE(nvtxRangePush("initialize source vectors");)
  for(int i=0; i<N; i++){
    vector_a[i] = rand() % 100;
    vector_b[i] = rand() % 100;
  }
  PROFILE(PROFILE(nvtxRangePop());)
  
  // set vector_a and vector_b as read mostly
  // COPY vector_a and vector_b to gpu
  cudaMemAdvise(vector_a, vector_size, cudaMemAdviseSetReadMostly, gpu_id);
  cudaMemAdvise(vector_b, vector_size, cudaMemAdviseSetReadMostly, gpu_id);
  cudaMemPrefetchAsync(vector_a, vector_size, gpu_id);
  cudaMemPrefetchAsync(vector_b, vector_size, gpu_id);

  // initialize kernel configuration
  // number of kernels per block
  int NUM_THREADS_PER_BLOCK = 1 << 10;

  // number of blocks per grid
  int NUM_BLOCKS_PER_GRID = N % NUM_THREADS_PER_BLOCK == 0 ?
                            N / NUM_THREADS_PER_BLOCK :
                            N / NUM_THREADS_PER_BLOCK + 1;

  // launch kernel
  PROFILE(
    std::cout << "Launch Kernel: " 
      << NUM_THREADS_PER_BLOCK << " threads per block, " 
      << NUM_BLOCKS_PER_GRID << " blocks per grid" 
      << std::endl;
    nvtxRangePush("start kernel");
  )
  vectorAdd<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(vector_a, vector_b, vector_c, N);
  
  cudaDeviceSynchronize();
  
  PROFILE(nvtxRangePop();)

  // migrate vector_a, vector_b and vector_c to cpu side
  // cudaMemPrefetchAsync(vector_a, vector_size, cudaCpuDeviceId);
  // cudaMemPrefetchAsync(vector_b, vector_size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(vector_c, vector_size, cudaCpuDeviceId);

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

