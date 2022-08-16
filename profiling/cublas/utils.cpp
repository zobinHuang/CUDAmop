#include <cassert>
#include <math.h>

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