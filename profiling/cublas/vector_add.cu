/*!
 * \file    vector_add.cu
 * \brief   cuBLAS version of vector addition
 * \author  Zhuobin Huang
 * \date    Aug. 16, 2022
 */

#include <iostream>
#include <cublas_v2.h>
#include <nvToolsExt.h>
#include <profile.h>

extern void verifyVectorAdditionResult(
    float *vector_a, 
    float *vector_b, 
    float *vector_c,
    float factor,
    int d
);

extern void vectorRandomInit(
  float *vector,
  int d
);

int main(){
    // initialize constants
    constexpr int N = 1 << 16;
    constexpr int unit_size = sizeof(float);
    constexpr int vector_size = N*unit_size;

    // declare vector pointers
    float *vector_a, *vector_b, *vector_c;
    float *d_vector_a, *d_vector_b;

    // create vector in host memory
    PROFILE(nvtxRangePush("allocate host memory for three vectors");)
    vector_a = (float*)malloc(vector_size);
    vector_b = (float*)malloc(vector_size);
    vector_c = (float*)malloc(vector_size);
    PROFILE(nvtxRangePop();)

    // create vector in device memory
    PROFILE(nvtxRangePush("allocate device memory for two source vectors");)
    cudaMalloc(&d_vector_a, vector_size);
    cudaMalloc(&d_vector_b, vector_size);
    PROFILE(nvtxRangePop();)

    // initilize vectors
    PROFILE(nvtxRangePush("initilize two source vector with random numbers");)
    vectorRandomInit(vector_a, N);
    vectorRandomInit(vector_b, N);
    PROFILE(nvtxRangePop();)

    // create and initialize a new cuBLAS context
    PROFILE(nvtxRangePush("create cuBLAS context");)
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    PROFILE(nvtxRangePop();)

    // copy two source vectors to device memory
    PROFILE(nvtxRangePush("copy source vectors to the device");)
    cublasSetVector(N, unit_size, vector_a, 1, d_vector_a, 1);
    cublasSetVector(N, unit_size, vector_b, 1, d_vector_b, 1);
    PROFILE(nvtxRangePop();)

    // launch saxpy kernel (single precision)
    // d_vector_b = alpha*d_vector_a + d_vector_b
    const float alpha = 2.0f;
    PROFILE(nvtxRangePush("launch cuBLAS Saxpy kernel");)
    cublasSaxpy_v2(handle, N, &alpha, d_vector_a, 1, d_vector_b, 1);
    PROFILE(nvtxRangePop();)

    // copy the result to host
    PROFILE(nvtxRangePush("copy result from device to host");)
    cublasGetVector(N, unit_size, d_vector_b, 1, vector_c, 1);
    PROFILE(nvtxRangePop();)

    // verify the result
    PROFILE(nvtxRangePush("verify calculation result");)
    verifyVectorAdditionResult(vector_a, vector_b, vector_c, alpha, N);
    PROFILE(nvtxRangePop();)

    // clean up cublas context
    PROFILE(nvtxRangePush("clean up cublas context");)
    cublasDestroy_v2(handle);
    PROFILE(nvtxRangePop();)

    // free host memory
    PROFILE(nvtxRangePush("free host memory");)
    free(vector_a);
    free(vector_b);
    free(vector_c);
    PROFILE(nvtxRangePop();)

    // free device memory
    PROFILE(nvtxRangePush("free device memory");)
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    PROFILE(nvtxRangePop();)

    std::cout << "Get correct vector addition result!" << std::endl;

    return 0;
}





