#ifndef _MATH_UTIL_CUH_
#define _MATH_UTIL_CUH_

/*!
 * \brief   returns log2(n) for a positive integer type
 * \param   n   the given positive integer
 * \param   p   number of incursive calling (internal used)
*/
template <typename T>
constexpr int IntegerLog2(T n, int p = 0) {
  return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
}

/*!
 * \brief   judge whether the integer is power of 2
 * \param   v   the given integer
 * \return  the judgement result
*/
template <typename T>
constexpr bool IntegerIsPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

#endif