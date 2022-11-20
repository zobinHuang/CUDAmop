/*!
 * \file    sum_reduction.cu
 * \brief   Operator for sum reduction
 * \author  Zhuobin Huang
 * \date    July 31, 2022
 */

#include <stdio.h>

/*!
 * \brief [CUDA Kernel] Conduct naive sum reduction (per thread block)
 * \param soure_array       source array
 * \param dest_array        destination array
 */
__global__ void sumReduction(
    int *soure_array, 
    int *dest_array){
  // obtain shared memory (through dynamic allcated)
  extern __shared__ int partial_sum[];

  // calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // load element into shared memory (current thread block)
  partial_sum[threadIdx.x] = soure_array[tid];
  __syncthreads();

  // iterate all number by base of 2
  for(int i=1; i<blockDim.x; i*=2){
    // calculation
    if(threadIdx.x % (2*i) == 0){ // note: cause warp divergence here
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x+i];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
      dest_array[blockIdx.x] = partial_sum[0];
  }
}

/*!
 * \brief [CUDA Kernel] Conduct sum reduction (per thread block)
 *        without warp divergence
 * \param soure_array       source array
 * \param dest_array        destination array
 */
__global__ void nonDivergenceSumReduction(
    int *soure_array, 
    int *dest_array){
  // obtain shared memory (through dynamic allcated)
  extern __shared__ int partial_sum[];

  // calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // load element into shared memory (current thread block)
  partial_sum[threadIdx.x] = soure_array[tid];
  __syncthreads();

  // iterate all number by base of 2
  for(int i=1; i<blockDim.x; i*=2){
    // get array index for current thread
    int index = 2*i*threadIdx.x;

    // calculation
    if(index<blockDim.x){
      partial_sum[index] += partial_sum[index+i];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
      dest_array[blockIdx.x] = partial_sum[0];
  }
}