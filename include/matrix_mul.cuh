/*!
 * \file    vector_addition.h
 * \brief   Function prototype for matrix multiplication
 * \author  Zhuobin Huang
 * \date    July 28, 2022
 */

#ifndef _MATRIX_MUL_H_
#define _MATRIX_MUL_H_

#include<stdint.h>

__global__ void squareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int d
);

__global__ void alignedSquareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int d
);

__global__ void tiledSquareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    const int tile_size,
    const int d
);

#endif
