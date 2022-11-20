/*!
 * \file    spmv.cu
 * \brief   Operator for sparse matrix-vector multiplication (SpMV)
 * \author  Zhuobin Huang
 * \date    Nov. 20, 2022
 */

#include <stdio.h>
#include <spmv.cuh>
#include <register.h>

/*!
 * \brief [CUDA Kernel] Conduct naive SpMV (one thread per row)
 * \param n_rows        number of rows in the sparse matrix
 * \param col_ids       CSR col indices
 * \param row_ptr       CSR row pointers
 * \param data          CSR data
 * \param x             the multiplied vector
 * \param y             the result vector
 */
__global__ void naiveCSRSpMV(
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

