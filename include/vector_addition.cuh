/*!
 * \file    vector_addition.h
 * \brief   Function prototype for vector addition
 * \author  Zhuobin Huang
 * \date    July 25, 2022
 */

#ifndef _VECTOR_ADDITION_H_
#define _VECTOR_ADDITION_H_

#include<stdint.h>

__global__ void vectorAdd(
    const int *__restrict vector_a, 
    const int *__restrict vector_b, 
    int *__restrict vector_c, 
    int d
);

__global__ void vectorAdd(
    int *vector_a, 
    int *vector_b, 
    int *vector_c, 
    int d
);

#endif