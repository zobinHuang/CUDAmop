#ifndef _RADIX_TYPE_TRANSLATOR_CUH_
#define _RADIX_TYPE_TRANSLATOR_CUH_

#include <stdint.h>

namespace cudamop {

namespace select {

namespace radix {

/*!
 * \note don't change this type!
 */
using RadixType = uint32_t;

template <typename T>
struct TypeTranslator {
    // static inline __device__ RadixType convert(T v){ static_assert(false, "not implemented"); }
    // static inline __device__ T deconvert(RadixType v){ static_assert(false, "not implemented"); }
};

/*!
 * \brief   converts a float to an integer representation with the same
 *          sorting, and vice versa; i.e., for floats f1, f2:
 *              if f1 < f2 then convert(f1) < convert(f2).
 *          we use this to enable radix selection of floating-point values,
 *          this also gives a relative order for NaNs, but that's ok, as they
 *          will all be adjacent
 *          neg inf: signbit=1 exp=ff fraction=0 --> radix = 0 00 ff..; 
 *          pos inf: signbit=0 exp=ff fraction=0 --> radix = 1 ff 00..; 
 *          pos nan: signbit=0 exp=ff fraction>0 --> radix = 1 ff x>0; 
 *          neg nan: signbit=1 exp=ff fraction>0 --> radix = 0 00 x<ff...
 */
template <>
struct TypeTranslator<float> {
    static inline __device__ RadixType convert(float v) {
        RadixType x = __float_as_int(v);
        RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (v == v) ? (x ^ mask) : 0xffffffff;
    }
    
    static inline __device__ float deconvert(RadixType v) {
        RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

        return __int_as_float(v ^ mask);
    }
};

/*!
 * \brief   converts a uint8_t to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<uint8_t> {
  static inline __device__ RadixType convert(uint8_t v) { return v; }
  static inline __device__ uint8_t deconvert(RadixType v) { return v; }
};

/*!
 * \brief   converts a int8_t to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<int8_t> {
  static inline __device__ RadixType convert(int8_t v) { return 128u + v; }
  static inline __device__ int8_t deconvert(RadixType v) { return v - 128; }
};

/*!
 * \brief   converts a int16_t to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<int16_t> {
  static inline __device__ RadixType convert(int16_t v) {
    static_assert(sizeof(short) == 2, "");
    return 32768u + v;
  }

  static inline __device__ int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

/*!
 * \brief   converts a int32_t to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<int32_t> {
  static inline __device__ RadixType convert(int32_t v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

/*!
 * \brief   converts a int64_t to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<int64_t> {
  static inline __device__ RadixType convert(int64_t v) {
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

/*!
 * \brief   converts a double to an integer representation with the same
 *          sorting, and vice versa
 */
template <>
struct TypeTranslator<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

} // namespace radix

} // namespace select

} // namespace cudamop

#endif