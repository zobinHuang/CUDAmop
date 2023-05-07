/*!
 * \file    tiled.cu
 * \brief   Cache titled version of matrix multiplication
 * \author  Zhuobin Huang
 * \date    July 29, 2022
 */

#include <iostream>
#include <vector>
#include <nvToolsExt.h>

#include <profile.h>
#include <matrix_mul.cuh>

void verifyMatrixMultiplicationResult(
  std::vector<int> &matrix_A, 
  std::vector<int> &matrix_B, 
  std::vector<int> &matrix_C,
  int N
);

int main(){
  // initialize constants
  constexpr int N = 1 << 10;
  constexpr int unit_size = sizeof(int);
  constexpr int matrix_size = N*N*unit_size;

  // create matrices (flat vectors) in host memory
  PROFILE(nvtxRangePush("allocate host memory for three matrices");)
  std::vector<int> matrix_A;
  matrix_A.reserve(N*N);
  std::vector<int> matrix_B;
  matrix_B.reserve(N*N);
  std::vector<int> matrix_C;
  matrix_C.reserve(N*N);
  PROFILE(nvtxRangePop();)

  // initialize random value for each source matrix
  PROFILE(nvtxRangePush("initialize two source matrices with random numbers");)
  for (int i=0; i<N*N; i++){
      matrix_A.push_back(rand()%100);
      matrix_B.push_back(rand()%100);
  }
  PROFILE(nvtxRangePop();)

  // allocate memory space on device
  PROFILE(nvtxRangePush("allocate device memory for three matrices");)
  int *d_matrix_A, *d_matrix_B, *d_matrix_C;
  cudaMalloc(&d_matrix_A, matrix_size);
  cudaMalloc(&d_matrix_B, matrix_size);
  cudaMalloc(&d_matrix_C, matrix_size);
  PROFILE(nvtxRangePop();)

  // copy data from host memory to device memory
  PROFILE(nvtxRangePush("copy matrices from host to device memory");)
  cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix_C, matrix_C.data(), matrix_size, cudaMemcpyHostToDevice);
  PROFILE(nvtxRangePop();)

  // initialize kernel configuration
  // number of kernels per block (one dimension)
  int NUM_THREADS_PER_BLOCK = 1 << 5;

  // number of blocks in the grid (one dimension)
  int NUM_BLOCKS =  N % NUM_THREADS_PER_BLOCK == 0 ?
                    N / NUM_THREADS_PER_BLOCK :
                    N / NUM_THREADS_PER_BLOCK + 1;

  // use dim3 struct for block and grid dimensions
  dim3 threads(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);
  dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

  // obtain shared memory size for each thread block
  int shared_memory_size = 2*NUM_THREADS_PER_BLOCK*NUM_THREADS_PER_BLOCK*unit_size;

  // launch kernel
  PROFILE(
      // std::cout << "Launch Kernel: " 
      // << NUM_THREADS_PER_BLOCK << " threads per block, " 
      // << NUM_BLOCKS << " blocks in the grid" 
      // << std::endl;
      nvtxRangePush("start kernel");
  )
  tiledSquareMatrixMul<<<blocks, threads, shared_memory_size>>>(
                d_matrix_A, d_matrix_B, d_matrix_C, NUM_THREADS_PER_BLOCK, N);
  // tiledSquareMatrixMul<<<blocks, threads>>>(
  //               d_matrix_A, d_matrix_B, d_matrix_C, NUM_THREADS_PER_BLOCK, N);
  PROFILE(
      // cudaDeviceSynchronize is not necessary as following cudaMemcpy would
      // act as synchronization barrier, used here for profiling log
      cudaDeviceSynchronize();
      nvtxRangePop();
  )

  // copy result back to host memory
  PROFILE(nvtxRangePush("copy matrix from device to host memory");)
  cudaMemcpy(matrix_C.data(), d_matrix_C, matrix_size, cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // verify result
  PROFILE(nvtxRangePush("verify result");)
  verifyMatrixMultiplicationResult(matrix_A, matrix_B, matrix_C, N);
  PROFILE(nvtxRangePop();)

  // free device memory
  PROFILE(nvtxRangePush("free device memory");)
  cudaFree(d_matrix_A);
  cudaFree(d_matrix_B);
  cudaFree(d_matrix_C);
  PROFILE(nvtxRangePop();)

  std::cout << "Get correct matrix multiplication result!" << std::endl;

  return 0;
}