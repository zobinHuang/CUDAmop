#ifndef _GPU_DEF_CUH_
#define _GPU_DEF_CUH_

static __device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
  return laneId;
}

static __device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

#endif