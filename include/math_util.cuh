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

#endif