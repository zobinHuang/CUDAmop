#ifndef _MATH_UTIL_CUH_
#define _MATH_UTIL_CUH_

namespace cudamop {

namespace utils {

/*!
 * \brief   returns log2(n) for a positive integer type
 * \param   n   the given positive integer
 * \param   p   number of incursive calling (internal used)
*/
template <typename T>
__device__ constexpr int IntegerLog2(T n, int p = 0) {
  return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
}

/*!
 * \brief   judge whether the integer is power of 2
 * \param   v   the given integer
 * \return  the judgement result
*/
template <typename T>
__device__ constexpr bool IntegerIsPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

/*!
 * \brief   computes ceil(a / b)
 * \param   a   value a
 * \param   b   value b
 * \return  ceil divide result
 */
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
__device__ __host__ T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/*!
 * \brief   computes ceil(a / b) * b; 
 *          i.e., rounds up `a` to the next highest multiple of b
 * \param   a   value a
 * \param   b   value b
 * \return  round up result
 */
template <typename T>
__device__ __host__ T roundUp(T a, T b) {
  return ceilDiv(a, b) * b;
}

} // namespace utils

} // namespace cudamop

#endif