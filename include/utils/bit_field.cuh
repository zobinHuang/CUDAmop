#ifndef _BITFIELD_CUH_
#define _BITFIELD_CUH_

#include <stdint.h>

namespace cudamop {

namespace utils {

/*!
 * \brief bit field utility functor
 */
template <typename T>
struct BitField {
  // static __device__ __host__ __forceinline__
  // unsigned int getBitField(unsigned int val, int pos, int len) {
  //   static_assert(false, "not implemented");
  // }

  // static __device__ __host__ __forceinline__
  // unsigned int setBitField(unsigned int val, unsigned int toInsert, int pos, int len) {
  //   static_assert(false, "not implemented");
  // }
};

/*!
 * \brief bit field utility functor for unsigned int
 */
template <>
struct BitField<unsigned int> {
  static __device__ __host__ __forceinline__
  unsigned int getBitField(unsigned int val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ __host__ __forceinline__
  unsigned int setBitField(unsigned int val, unsigned int toInsert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

/*!
 * \brief bit field utility functor for uint64_t
 */
template <>
struct BitField<uint64_t> {
  static __device__ __host__ __forceinline__
  uint64_t getBitField(uint64_t val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ __host__ __forceinline__
  uint64_t setBitField(uint64_t val, uint64_t toInsert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    uint64_t ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

} // namespace utils

} // namespace cudamop

#endif