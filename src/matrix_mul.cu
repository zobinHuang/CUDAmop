/*!
 * \file    matrix_mul.cu
 * \brief   Operator for matrix multiplication
 * \author  Zhuobin Huang
 * \date    July 28, 2022
 */

#include <matrix_mul.cuh>
#include <cassert>

/*!
 * \brief [CUDA Kernel] Conduct square matrix multiplication (A*B=C)
 * \param matrix_A  source matrix
 * \param matrix_B  source matrix
 * \param matrix_C  destination matrix
 * \param d         dimension of square matrix
 */
__global__ void squareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    int d){
  // check kernel shape
  assert(blockDim.x == blockDim.y);
  assert(gridDim.x == gridDim.y);

  // obtain corresponding row and column for current thread
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;

  // initialize destination element
  int dest_index = row_index*d+col_index;
  matrix_C[dest_index] = 0;

  // sum of product
  if(dest_index < d*d){
    for(int i=0; i<d; i++){
      matrix_C[dest_index] += matrix_A[row_index*d+i] * matrix_B[i*d+col_index];
    }
  }
}

/*!
 * \brief [CUDA Kernel] Conduct square matrix multiplication (A*B=C)
 *        based on cache tiling, make sure tile size is equal to 
 *        block size
 * \param matrix_A  source matrix
 * \param matrix_B  source matrix
 * \param matrix_C  destination matrix
 * \param d         dimension of square matrix
 */
__global__ void tiledSquareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    int tile_size,
    int d){
    // check kernel shape
    assert(blockDim.x == blockDim.y);
    assert(gridDim.x == gridDim.y);
    assert(tile_size == blockDim.x);
}