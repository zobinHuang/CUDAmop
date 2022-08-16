#include <iostream>
#include <cassert>
#include <math.h>

/*!
 * \brief Verify the squared matrix multiplication result from GPU
 *        matrix_d = (alpha*matrix_a) * matrix_b + (beta*matrix_c)
 * \param matrix_a      pointer to source matrix a
 * \param matrix_b      pointer to source matrix b
 * \param matrix_c      pointer to source matrix c
 * \param matrix_d      pointer to destination matrix
 * \param factor_alpha  the multiplied coefficient on matrix_a
 * \param factor_beta   the multiplied coefficient on matrix_c
 * \param epsilon       error tolerance
 * \param d             dimension of squared matrix
 */
void verifyCUBLASSgemmResult(
    float *matrix_a, 
    float *matrix_b, 
    float *matrix_c,
    float *matrix_d,
    const float factor_alpha,
    const float factor_beta,
    const float epsilon,
    int d
  ){
    float temp;
    // for each row in matrix_c
    for(int i=0; i<d; i++){
      // for each column in matrix_c
      for(int j=0; j<d; j++){
        temp = 0;
        // calculate actual result
        for(int k=0; k<d; k++){
          temp += factor_alpha*matrix_a[k*d+i]*matrix_b[j*d+k];
        }
        temp += factor_beta*matrix_c[j*d+i];
        assert(fabs(matrix_d[j*d+i]-temp) < epsilon);
      }
    }
}

/*!
 * \brief Verify the vector addition result from GPU
 * \param vector_a  pointer to source vector array
 * \param vector_b  pointer to source vector array
 * \param vector_c  pointer to destination vector array
 * \param factor    the multiplied coefficient on vector_a
 * \param d         dimension of vectors
 */
void verifyVectorAdditionResult(
    float *vector_a, 
    float *vector_b, 
    float *vector_c,
    float factor,
    int d
  ){
    for(int i=0; i<d; i++){
      assert(vector_c[i] == factor*vector_a[i]+vector_b[i]);
    }
}

/*!
 * \brief Initialize vector with random float numbers
 * \param vector    pointer to the vector
 * \param d         dimension of the vector
 */
void vectorRandomInit(
  float *vector,
  int d
){
  for(int i=0; i<d; i++){
    vector[i] = (float)(rand()%100);
  }
}