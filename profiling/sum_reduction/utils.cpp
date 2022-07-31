/*!
 * \file    utils.cpp
 * \brief   Utilization function set
 * \author  Zhuobin Huang
 * \date    July 31, 2022
 */

#include <iostream>
#include <numeric>
#include <cassert>
#include <vector>

void verifySumReductionResult(
    std::vector<int> &source_array,
    std::vector<int> &destination_array
  ){
  assert(
    destination_array[0] 
      == std::accumulate(begin(source_array), end(source_array), 0)
  );
}