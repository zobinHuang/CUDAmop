/*!
 * \file    utils.cpp
 * \brief   Utilization function set
 * \author  Zhuobin Huang
 * \date    July 27, 2022
 */

#include <iostream>
#include <cassert>
#include <vector>

/*!
 * \brief Verify the vector addition result from GPU
 * \param vector_a source vector
 * \param vector_b source vector
 * \param vector_c destination vector
 */
void verifyVectorAdditionResult(
    std::vector<int> &vector_a, 
    std::vector<int> &vector_b, 
    std::vector<int> &vector_c
  ){
    for(int i=0; i<vector_a.size(); i++){
      assert(vector_c[i] == vector_a[i]+vector_b[i]);
    }
}

/*!
 * \brief Verify the vector addition result from GPU
 * \param vector_a  pointer to source vector array
 * \param vector_b  pointer to source vector array
 * \param vector_c  pointer to destination vector array
 * \param d         dimension of vectors
 */
void verifyVectorAdditionResult(
    int *vector_a, 
    int *vector_b, 
    int *vector_c,
    int d
  ){
    for(int i=0; i<d; i++){
      assert(vector_c[i] == vector_a[i]+vector_b[i]);
    }
}