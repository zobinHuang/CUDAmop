/*!
 * \file    matrix_multiplication.cu
 * \brief   cuBLAS version of matrix multiplication
 * \author  Zhuobin Huang
 * \date    Aug. 16, 2022
 */

#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <nvToolsExt.h>
#include <profile.h>

extern void verifyCUBLASSgemmResult(
    float *matrix_a, 
    float *matrix_b, 
    float *matrix_c,
    float *matrix_d,
    const float factor_alpha,
    const float factor_beta,
    const float epsilon,
    int d
);

int main(){
  // initialize constants
  constexpr int N = 1 << 10;
  constexpr int unit_size = sizeof(float);
  constexpr int matrix_size = N*N*unit_size;
  
  // declare vector pointers
  float *matrix_a, *matrix_b, *matrix_c, *matrix_d;
  float *d_matrix_a, *d_matrix_b, *d_matrix_c;

  // create vector in host memory
  PROFILE(nvtxRangePush("allocate host memory for three matrices");)
  matrix_a = (float*)malloc(matrix_size);
  matrix_b = (float*)malloc(matrix_size);
  matrix_c = (float*)malloc(matrix_size);
  matrix_d = (float*)malloc(matrix_size);
  PROFILE(nvtxRangePop();)

  // create vector in device memory
  PROFILE(nvtxRangePush("allocate device memory for three matrices");)
  cudaMalloc(&d_matrix_a, matrix_size);
  cudaMalloc(&d_matrix_b, matrix_size);
  cudaMalloc(&d_matrix_c, matrix_size);
  PROFILE(nvtxRangePop();)

  // pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill matrix with random number with cuRAND on the device
  PROFILE(nvtxRangePush("initilize three source matrices with random numbers using cuRAND");)
  curandGenerateUniform(prng, d_matrix_a, N*N);
  curandGenerateUniform(prng, d_matrix_b, N*N);
  curandGenerateUniform(prng, d_matrix_c, N*N);
  PROFILE(nvtxRangePop();)

  // save the initial value of matrix c for verification after kernel finished
  PROFILE(nvtxRangePush("save the initial value of matrix c");)
  cudaMemcpy(matrix_c, d_matrix_c, matrix_size, cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // create and initialize a new cuBLAS context
  PROFILE(nvtxRangePush("create cuBLAS context");)
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  PROFILE(nvtxRangePop();)

  // launch sgemm kernel
  // (1) calculation:   d_matrix_c = (alpha*d_matrix_a) * d_matrix_b + (beta*d_matrix_c)
  // (2) matrix shape:  d_matrix_a (m X n), d_matrix_b (n X k), d_matrix_c (m X k)
  const float alpha = 2.0f;
  const float beta = 1.0f;
  PROFILE(nvtxRangePush("launch cuBLAS Sgemm kernel");)
  cublasSgemm_v2(
    handle,         // cuBLAS handler 
    CUBLAS_OP_N,    // indicate source matrix a is normal matrix, 
                    // and has no need to be transposed
    CUBLAS_OP_N,    // indicate source matrix b is normal matrix, 
                    // and has no need to be transposed
    N, N, N,        // matrix shape m, n, k
    &alpha,         // constant coefficient alpha
    d_matrix_a, N,  // source matrix a and its leading dimension
    d_matrix_b, N,  // source matrix b and its leading dimension
    &beta,          // constant coefficient beta
    d_matrix_c, N   // source matrix c and its leading dimension
  );
  PROFILE(nvtxRangePop();)

  // copy result from device to host
  PROFILE(nvtxRangePush("copy result from device to host");)
  cudaMemcpy(matrix_a, d_matrix_a, matrix_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(matrix_b, d_matrix_b, matrix_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(matrix_d, d_matrix_c, matrix_size, cudaMemcpyDeviceToHost);
  PROFILE(nvtxRangePop();)

  // verify result
  PROFILE(nvtxRangePush("verify matrix multiplication result");)
  const float epsilon = 0.01f;  // too strict might be failed
  verifyCUBLASSgemmResult(
    matrix_a,       // source matrix a
    matrix_b,       // source matrix b
    matrix_c,       // source matrix c
    matrix_d,       // destination matrix d
    alpha, beta,    // the multiplied coefficient alpha and beta
    epsilon,        // error tolerance
    N               // dimension of squared matrix
  );
  PROFILE(nvtxRangePop();)

  // free host memory
  PROFILE(nvtxRangePush("free host memory");)
  free(matrix_a);
  free(matrix_b);
  free(matrix_c);
  free(matrix_d);
  PROFILE(nvtxRangePop();)

  // free device memory
  PROFILE(nvtxRangePush("free device memory");)
  cudaFree(d_matrix_a);
  cudaFree(d_matrix_b);
  cudaFree(d_matrix_c);
  PROFILE(nvtxRangePop();)

  std::cout << "Get correct matrix multiplication result!" << std::endl;

  return 0;
}