/*!
 * \file    vector_addition.h
 * \brief   Function prototype for matrix multiplication
 * \author  Zhuobin Huang
 * \date    July 28, 2022
 */

#ifndef _MATRIX_MUL_H_
#define _MATRIX_MUL_H_

__global__ void squareMatrixMul(
    const int *matrix_A,
    const int *matrix_B,
    int *matrix_C,
    int d
);

#endif
