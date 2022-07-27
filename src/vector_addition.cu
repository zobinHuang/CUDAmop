/*!
 * \file    vector_addition.cu
 * \brief   Operator for basic version of vector addition
 * \author  Zhuobin Huang
 * \date    July 25, 2022
 */

#include <iostream>
#include <vector>
#include <vector_addition.cuh>

/*!
 * \brief [CUDA Kernel] Conduct Vector Adding (a+b=c)
 * \param vector_a  source vector
 * \param vector_b  source vector
 * \param vector_c  destination vector
 * \param d         dimension of vectors
 */
__global__ void vectorAdd(
    const int *__restrict vector_a, 
    const int *__restrict vector_b, 
    int *__restrict vector_c, 
    int d){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d){
    vector_c[tid] = vector_a[tid] + vector_b[tid];
  }
}