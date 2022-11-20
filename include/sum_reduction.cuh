/*!
 * \file    sum_reduction.h
 * \brief   Function prototype for sum reduction
 * \author  Zhuobin Huang
 * \date    July 31, 2022
 */

#ifndef _SUM_REDUCTION_H_
#define _SUM_REDUCTION_H_

#include<stdint.h>

__global__ void sumReduction(
    int *soure_array, 
    int *dest_array
);

__global__ void nonDivergenceSumReduction(
    int *soure_array, 
    int *dest_array
);

#endif