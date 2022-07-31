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
    const int d){
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
 *        with aligned memory access pattern
 * \param matrix_A  source matrix
 * \param matrix_B  source matrix (after transposed)
 * \param matrix_C  destination matrix
 * \param d         dimension of square matrix
 */
__global__ void alignedSquareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int d){
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
      matrix_C[dest_index] += matrix_A[row_index*d+i] * matrix_B[col_index*d+i];
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
 * \param tile_size size of each tile
 * \param d         dimension of square matrix
 */
__global__ void tiledSquareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int tile_size,
    const int d){
  // check kernel shape
  assert(blockDim.x == blockDim.y);
  assert(gridDim.x == gridDim.y);
  assert(tile_size == blockDim.x);
  
  // obtained shared memory area (dynamic allocated)
  extern __shared__ int tile[];
  int* tile_A = tile;
  int* tile_B = tile+tile_size*tile_size;

  // obtain global row and column index for current thread
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // obtain shared memory index for current thread
  int shared_tile_index = threadIdx.y*blockDim.x+threadIdx.x;

  int tmp = 0;

  // traverse all tiles
  for(int i=0; i<d; i += tile_size){
    // load element into shared memory
    tile_A[shared_tile_index] = matrix_A[row_index*d+threadIdx.x+i];
    tile_B[shared_tile_index] = matrix_B[threadIdx.y*d+col_index+i*d];

    // wait for both tiles to be loaded by threads in current CLB
    __syncthreads();

    // computation
    for(int j=0; j<tile_size; j++){
      tmp += tile_A[threadIdx.y*blockDim.x+j]*tile_B[j*blockDim.x+threadIdx.x];
    }

    // wait for all threads finish computation for current tiles
    // before loading in new one
    __syncthreads();
  }

  // write result to global memory
  matrix_C[row_index*d+col_index] = tmp;
}