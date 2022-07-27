#ifndef _VECTOR_ADDITION_H_
#define _VECTOR_ADDITION_H_

#include <vector>

__global__ void vectorAdd(
    const int *__restrict vector_a, 
    const int *__restrict vector_b, 
    int *__restrict vector_c, 
    int d
);

#endif