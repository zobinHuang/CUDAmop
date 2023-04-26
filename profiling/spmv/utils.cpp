/*!
 * \file    utils.cpp
 * \brief   Utilization function set
 * \author  Zhuobin Huang
 * \date    Nov.20 2022
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <register.h>
#include <sys/time.h>

float generateRandomValue(){
    if( 
        typeid(float) == typeid(short) or 
        typeid(float) == typeid(int) or 
        typeid(float) == typeid(long) or 
        typeid(float) == typeid(unsigned short) or
        typeid(float) == typeid(unsigned int) or
        typeid(float) == typeid(unsigned long)
    ) {
        return rand()%100;
    } else if (
        typeid(float) == typeid(float) or 
        typeid(float) == typeid(double)
    ) {
        return static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    } else {
        std::cout << "Unknown type name " << typeid(float).name() << std::endl;
        return static_cast<float>(1);
    }
}

void SpMVSeq(
    const uint64_t row_size,
    const uint64_t num_row,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data,
    std::vector<float> &x,
    std::vector<float> &result
){
    for(uint64_t row=0; row<num_row; row++){
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row+1];
        float sum = 0;
        for(uint64_t i=row_start; i<row_end; i++){
            sum += data[i] * x[col_ids[i]];
        }
        result[row] = sum;
    }
}

void verifySpMVResult(
    const uint64_t row_size,
    const uint64_t num_row,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data,
    std::vector<float> &x,
    std::vector<float> &result
){
    std::vector<float> correct_result(num_row, 0);
    
    for(uint64_t row=0; row<num_row; row++){
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row+1];
        float sum = 0;
        for(uint64_t i=row_start; i<row_end; i++){
            sum += data[i] * x[col_ids[i]];
        }
        correct_result[row] = sum;
    }

    for(uint64_t row=0; row<num_row; row++){
        // std::cout << "result: " << result[row] << "; correct result: " << correct_result[row] << std::endl;
        assert(abs(result[row]-correct_result[row]) <= 0.00001);
    }
}

/*!
 * \brief Generate random sparse matrix in CSR format
 * \param row_size  number of elements per row
 * \param elem_cnt  number of elements in the matrix
 * \param nnz       number of non-zero elements
 * \param col_ids   CSR column indices
 * \param row_ptr   CSR row pointers
 * \param data      CSR data
 */
void generateRandomCSR(
    const uint64_t row_size,
    const uint64_t elem_cnt,    
    const uint64_t nnz,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data
){
    // generate random sparse matrix
    std::vector<float> temp_array(elem_cnt, 0);
    for (uint64_t i=0; i<nnz; i++) {
        uint64_t index = (uint64_t) (elem_cnt * ((double) rand() / (RAND_MAX + 1.0)));
        temp_array[index] = generateRandomValue();
    }

    // assert elem_cnt is divided by row_size
    assert(elem_cnt%row_size == 0);

    // convert to CSR format
    uint64_t n_rows = elem_cnt/row_size;
    uint64_t nnz_count = 0;
    row_ptr.push_back(0);
    for(uint64_t row=0; row<n_rows; row++){
        uint64_t nnz_row = 0;
        for(uint64_t col=0; col<row_size; col++){
            if(temp_array[row*row_size+col] != 0){
                nnz_row += 1;
                col_ids.push_back(col);
                data.push_back(temp_array[row*row_size+col]);
            }
        }
        nnz_count += nnz_row;
        row_ptr.push_back(nnz_count);
    }
}