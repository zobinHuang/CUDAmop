/*!
 * \file    spmv.cuh
 * \brief   Function prototype for sparse matrix-vector multiplication (SpMV)
 * \author  Zhuobin Huang
 * \date    Nov.20 2022
 */

#ifndef _SPMV_H_
#define _SPMV_H_

#include<stdint.h>

__global__ void CSRSpMVScalar(
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y
);

__global__ void CSRSpMVVector (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y
);

#endif