/*!
 * \file    verify.cpp
 * \brief   Functions for verify matrix multiplication result
 * \author  Zhuobin Huang
 * \date    July 28, 2022
 */

#include <iostream>
#include <cassert>
#include <vector>

/*!
 * \brief Verify the matrix multiplication result from GPU
 * \param matrix_A  source matrix
 * \param matrix_B  source matrix
 * \param matrix_C  destination matrix
 * \param N         dimension of matrix 
 */
void verifyMatrixMultiplicationResult(
    std::vector<int> &matrix_A, 
    std::vector<int> &matrix_B, 
    std::vector<int> &matrix_C,
    int N
  ){
    // for every row of matrix_C
    for(int i=0; i<N; i++){
        // for every column of matrix_C
        for(int j=0; j<N; j++){
            // calculate correct result
            int tmp=0;
            for(int k=0; k<N; k++){
                tmp += matrix_A[N*i+k]*matrix_B[N*k+j];
            }
            // assertion
            assert(tmp == matrix_C[i*N+j]);
        }
    }
}