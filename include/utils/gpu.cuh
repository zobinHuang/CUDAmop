#ifndef _GPU_DEF_CUH_
#define _GPU_DEF_CUH_

namespace cudamop {

namespace utils {

/*!
 * \brief   get the lane id within current warp
 * \return  the lane id within current warp
 */
static __device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
  return laneId;
}

/*!
 * \brief   get the lane mask which cover the threads before current
 *          thread within the warp
 * \return  the lane mask
 */
static __device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

/*!
 * \brief   get the lane mask which cover the threads before current
 *          thread and current thread within the warp
 * \return  the lane mask
 */
__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

/*!
 * \brief   get the active mask of threads within current block
 * \return  the active mask of threads within current block
 */
static __device__ __forceinline__ unsigned int getActiveMask() {
  return __activemask();
}

/*!
 * \brief   get the vote result within current warp based on given mask
 * \param   predicate the vote result of current thread
 * \param   mask      the mask to specify which threads within the warp 
 *                    are participating voting
 * \return  number of threads that vote positive within current warp
 */
static __device__ __forceinline__ unsigned int getWarpBallot(int predicate, unsigned int mask=0xffffffff){
  return __ballot_sync(mask, predicate);
}

/*!
 * \brief   get the linear block index
 * \return  the linear block index
 */
template <typename index_t>
__device__ __forceinline__ index_t getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
      blockIdx.x;
}

} // namespace utils

} // namespace cudamop

#endif