/*!
 * \file    spmv.cu
 * \brief   Basic version of SpMV
 * \author  Zhuobin Huang
 * \date    Nov.20, 2022
 */

#include <iostream>
#include <vector>
#include <nvToolsExt.h>

#include <profile.h>
#include <spmv.cuh>

float generateRandomValue();

void generateRandomCSR(
    const uint64_t row_size,
    const uint64_t elem_cnt,    
    const uint64_t nnz,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data
);

void SpMVSeq(
    const uint64_t row_size,
    const uint64_t num_row,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data,
    std::vector<float> &x,
    std::vector<float> &result
);

void verifySpMVResult(
    const uint64_t row_size,
    const uint64_t num_row,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &data,
    std::vector<float> &x,
    std::vector<float> &result
);

int main(){
    // initialize constants
    constexpr uint64_t row_size = 1 << 20;
    constexpr uint64_t num_row = 1 << 10;
    uint64_t elem_cnt = row_size*num_row;
    uint64_t nnz = (uint64_t)((double)elem_cnt*0.5);
    uint64_t unit_size = sizeof(float);

    // create vector in host memory
    PROFILE(nvtxRangePush("allocate host memory for vectors");)
    std::vector<uint64_t> col_ids;      // CSR col indices
    std::vector<uint64_t> row_ptr;      // CSR row pointers
    std::vector<float> data;            // CSR data
    std::vector<float> x;               // the multiplied vector
    x.reserve(row_size);
    std::vector<float> y;               // the result vector
    y.reserve(num_row);
    PROFILE(nvtxRangePop();)

    // initialize random value
    PROFILE(nvtxRangePush("initialize matrix and vector with random numbers");)
    generateRandomCSR(row_size, elem_cnt, nnz, col_ids, row_ptr, data);
    for (int i=0; i<row_size; i++){
        x.push_back(generateRandomValue());
    }
    PROFILE(nvtxRangePop();)

    // allocate memory space on device
    PROFILE(nvtxRangePush("allocate memory space on device");)
    uint64_t *d_col_ids, *d_row_ptr;
    float *d_data, *d_x, *d_y;
    cudaMalloc(&d_col_ids,  sizeof(uint64_t)*col_ids.size());
    cudaMalloc(&d_row_ptr,  sizeof(uint64_t)*row_ptr.size());
    cudaMalloc(&d_data,     unit_size*data.size());
    cudaMalloc(&d_x,        unit_size*x.size());
    cudaMalloc(&d_y,        unit_size*num_row);
    PROFILE(nvtxRangePop();)

    // copy data from host memory to device memory
    PROFILE(nvtxRangePush("copy data from host to device memory");)
    cudaMemcpy(d_col_ids, col_ids.data(), sizeof(uint64_t)*col_ids.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(uint64_t)*row_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data.data(), unit_size*data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), unit_size*x.size(), cudaMemcpyHostToDevice);
    PROFILE(nvtxRangePop();)

    // initialize kernel configuration
    // number of kernels per block
    int NUM_THREADS_PER_BLOCK = 1 << 8;

    // number of blocks in the grid
    int NUM_BLOCKS =  num_row % NUM_THREADS_PER_BLOCK == 0 ?
                        num_row / NUM_THREADS_PER_BLOCK :
                        num_row / NUM_THREADS_PER_BLOCK + 1;

    // launch naive kernel
    PROFILE(
        // std::cout << "Launch Kernel: " 
        // << NUM_THREADS_PER_BLOCK << " threads per block, " 
        // << NUM_BLOCKS << " blocks in the grid" 
        // << std::endl;
        nvtxRangePush("launch naive kernel");
    )
    CSRSpMVScalar<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(num_row, d_col_ids, d_row_ptr, d_data, d_x, d_y);
    PROFILE(
        // cudaDeviceSynchronize is not necessary as following cudaMemcpy would
        // act as synchronization barrier, used here for profiling log
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
        nvtxRangePop();
    )

    // copy result back to host memory
    PROFILE(nvtxRangePush("copy vector from device to host memory");)
    cudaMemcpy(y.data(), d_y, unit_size*num_row, cudaMemcpyDeviceToHost);
    PROFILE(nvtxRangePop();)

    // verify result
    verifySpMVResult(row_size, num_row, col_ids, row_ptr, data, x, y);

    std::vector<float> result(num_row, 0);
    PROFILE(nvtxRangePush("sequence implementation");)
    SpMVSeq(row_size, num_row, col_ids, row_ptr, data, x, result);
    PROFILE(nvtxRangePop();)

    // free device memory
    PROFILE(nvtxRangePush("free device memory");)
    cudaFree(d_col_ids);
    cudaFree(d_row_ptr);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);
    PROFILE(nvtxRangePop();)

    std::cout << "Get correct SpMV result!" << std::endl;

    return 0;
}