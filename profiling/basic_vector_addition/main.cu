/*!
 * \file    main.cu
 * \brief   Basic version of vector addition
 * \author  Zhuobin Huang
 * \date    July 25, 2022
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <nvToolsExt.h>

#include <profile.h>
#include <vector_addition.cuh>

/*!
 * \brief verify the vector addition result from GPU
 * \param vector_a source vector
 * \param vector_b source vector
 * \param vector_c destination vector
 */
void verifyResult(
    std::vector<int> &vector_a, 
    std::vector<int> &vector_b, 
    std::vector<int> &vector_c){
  for(int i=0; i<vector_a.size(); i++){
    std::cout << "[i]=" << i 
      << ", vector_a= " << vector_a[i]
      << ", vector_b= " << vector_b[i]
      << ", vector_c= " << vector_c[i]
      << std::endl;
    assert(vector_c[i] == vector_a[i]+vector_b[i]);
  }
}

int main(){
  // initialize constants
  constexpr int N = 1 << 16;
  constexpr int unit_size = sizeof(int);
  constexpr int vector_size = N*unit_size;

  // create vector in host memory
  PROFILE(nvtxRangePush("allocate host memory for three vectors"));
  std::vector<int> vector_a;
  vector_a.reserve(N);
  std::vector<int> vector_b;
  vector_b.reserve(N);
  std::vector<int> vector_c;
  vector_c.reserve(N);
  PROFILE(nvtxRangePop());
  
  // initialize random value for each source vector
  PROFILE(nvtxRangePush("initialize two source vectors with random numbers"));
  for (int i=0; i<N; i++){
    vector_a.push_back(rand()%100);
    vector_b.push_back(rand()%100);
  }
  PROFILE(nvtxRangePop());

  // allocate memory space on device
  PROFILE(nvtxRangePush("allocate device memory for three vectors"));
  int *d_vector_a, *d_vector_b, *d_vector_c;
  cudaMalloc(&d_vector_a, vector_size);
  cudaMalloc(&d_vector_b, vector_size);
  cudaMalloc(&d_vector_c, vector_size);
  PROFILE(nvtxRangePop());

  // copy data from host memory to device memory
  PROFILE(nvtxRangePush("copy vectors from host to device memory"));
  cudaMemcpy(d_vector_a, vector_a.data(), vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_b, vector_b.data(), vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_c, vector_c.data(), vector_size, cudaMemcpyHostToDevice);
  PROFILE(nvtxRangePop());

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
    nvtxRangePush("start kernel")
  );
  vectorAdd<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK>>>(d_vector_a, d_vector_b, d_vector_c, vector_size);
  PROFILE(
    cudaDeviceSynchronize();
    nvtxRangePop();
  );
  
  // copy result back to host memory
  PROFILE(nvtxRangePush("copy vectors from device to host memory"));
  cudaMemcpy(vector_c.data(), d_vector_c, vector_size, cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop());

  // verify result
  PROFILE(nvtxRangePush("verify result"));
  verifyResult(vector_a, vector_b, vector_c);
  PROFILE(nvtxRangePop());

  // free device memory
  PROFILE(nvtxRangePush("free device memory"));
  cudaFree(d_vector_a);
  cudaFree(d_vector_b);
  cudaFree(d_vector_c);
  PROFILE(nvtxRangePop());

  std::cout << "Get correct vector addition result!" << std::endl;

  return 0;
}