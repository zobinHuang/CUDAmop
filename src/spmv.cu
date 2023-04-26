/*!
 * \file    spmv.cu
 * \brief   Operator for sparse matrix-vector multiplication (SpMV)
 * \author  Zhuobin Huang
 * \date    Nov. 20, 2022
 */

#include <stdio.h>
#include <spmv.cuh>
#include <register.h>

#define FULL_WARP_MASK 0xffffffff

/*!
 * \brief [CUDA Kernel] Conduct SpMV-Scalar (one thread per row)
 * \param n_rows        number of rows in the sparse matrix
 * \param col_ids       CSR col indices
 * \param row_ptr       CSR row pointers
 * \param data          CSR data
 * \param x             the multiplied vector
 * \param y             the result vector
 */
__global__ void CSRSpMVScalar(
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y
){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; // one thread per row
    if (row < n_rows) {
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row+1];

        float sum = 0;
        for(uint64_t i=row_start; i<row_end; i++){
            sum += data[i] * x[col_ids[i]];
        }
        
        y[row] = sum;
    }
}

__device__ float warp_reduce(float val){
    for(uint64_t i=warpSize/2; i>0; i/= 2){
        val += __shfl_down_sync(FULL_WARP_MASK, val, i);
    }
    return val;
}

__global__ void CSRSpMVVector (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y
){
    const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t warp_id = thread_id / 32;
    const uint64_t lane = thread_id % 32;
    const uint64_t row_id = warp_id;

    float sum = 0;
    if (row_id < n_rows) {
        const unsigned int row_start = row_ptr[row_id];
        const unsigned int row_end = row_ptr[row_id+1];
        
        for(uint64_t i=row_start+lane; i<row_end; i+=32){
            sum += data[i] * x[col_ids[i]];
        }
    }

    sum = warp_reduce(sum);

    if(lane == 0 && row_id < n_rows){
        y[row_id] = sum;
    }   
}