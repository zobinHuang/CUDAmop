#ifndef _RADIX_SELECT_CUH_
#define _RADIX_SELECT_CUH_

#include <stdint.h>

#include <utils/math.cuh>
#include <utils/bit_field.cuh>

#include "radix_type_translator.cuh"

namespace cudamop {

namespace select {

namespace radix {

/*!
 * \brief   counting number of elements that hold each radix digit, 
 *          under the constrition of disired prefix
 * \tparam  V               data type 
 * \tparam  RadixSize       number of values based on RadixBits (2^RadixBits)
 * \tparam  RadixBits       over what radix we are selecting values
 * \param   data            the input elements
 * \param   n               length of data
 * \param   counts          counters that save the number of elements that hold 
 *                          each radix digit
 * \param   smem            shared memory area for internal usage, the length is
 *                          sizeof(size_t) * RadixSize
 * \param   radixDigitPos   start position of radix digit (in binary), 
 *                          masking length is RadixBits
 * \param   desired         desired prefix
 * \param   desiredMask     mask of the desired prefix
 */
template<typename V, size_t RadixSize, size_t RadixBits>
__device__ void countRadixUsingMask(
    V *data, size_t n, unsigned int counts[RadixSize], unsigned int *smem, 
    size_t radixDigitPos, RadixType desired, RadixType desiredMask
){
    size_t i, j;
    
    // clear out per-thread counts from previous round
    for(i=0; i<RadixSize; i++){
        counts[i] = 0;
    }

    // clear shared memory counts from previous round
    if(threadIdx.x < RadixSize){
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    /*!
     * \note    the following for loop record #elements for each radix digit within each warp, 
     *          where each element also satisfy hasing the desired prefix
     */
    for(i=threadIdx.x; i<n; i+=blockDim.x){
        // convert the data from origin type to int representation for radix sorting
        RadixType val = TypeTranslator<V>::convert(data[i]);
        
        // we want this value only if the bits masked by desiredMask equals to desired
        bool hasVal = ((val & desiredMask) == desired);

        // extract radix digits with size of RadixBits
        RadixType digitsInRadix = cudamop::utils::BitField<RadixType>::getBitField(val, radixDigitPos, RadixBits);
    
        /*!
         * \note    counting the number of threads within this warp that has
         *          each digitsInRadix under desired prefix
         * \note    the following part is expected to be executed serially 
         *          warp-by-warp, so we don't need to use fence like 
         *          __threadfence_block to ensure the write to shared memory 
         *          will be feasible to other threads in this block, it already does!
         */
#pragma unroll
        for(j=0; j<RadixSize; j++){
            bool vote = hasVal && (digitsInRadix == j);

            /*
             * \note    some threads might be diverged as been returned in radixSelect,
             *          so we need to invoke __activemask() to filter out these threads
             *          while conducting __ballot_sync()
             * \ref     usage of following warp primitive: 
             *          https://stackoverflow.com/a/54055576/14377282
             */
            {
                unsigned int active_mask = cudamop::utils::getActiveMask();
                unsigned int warp_vote_res = cudamop::utils::getWarpBallot(vote, active_mask);
                counts[j] += __popc(warp_vote_res);
            }
        }
    } // for(i=threadIdx.x; i<n; i+=blockDim.x)
    
    /*!
     * \note    accumulate each warp's result into the share memory
     */
    if(cudamop::utils::getLaneId() == 0){
        for(i=0; i<RadixSize; i+=1){
            atomicAdd(&smem[i], counts[i]);
        }
    }
    __syncthreads();

    /*!
     * \note    for each thread, read in the total counts
     */
#pragma unroll
    for(i=0; i<RadixSize; i++){
        counts[i] = smem[i];
    }

    /*!
     * \note    we need to syncthreads here as the shared memory will be clear in the next round
     */
    __syncthreads();
}

/*!
 * \brief   over what radix we are selecting values, 
 *          digits are base-(2 ^ RADIX_BITS)
 */
constexpr int RADIX_BITS = 2;

/*
 * \brief   number of values based on RADIX_BITS (2^RADIX_BITS)
 */
constexpr int RADIX_SIZE = 1 << RADIX_BITS;

/*
 * \brief   full radix mask
 */
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

/*!
 * \brief   find the unique value `v` that matches the pattern
 *          ((v & desired) == desiredMask)
 * \tparam  V           data type
 * \param   smem        shared memory area for internal usage
 *                      (which is small as two slots with type V)
 * \param   data        the data which the unique v be in
 * \param   n           length of data
 * \param   desired     the desired pattern
 * \param   desiredMask the mask of the desired pattern
 * \return  the founded unique value
 */
template<typename V>
__device__ V findPattern(V *smem, V *data, size_t n, RadixType desired, RadixType desiredMask){
    size_t i;

    // we only use two slots with type V in the shared memory
    if(threadIdx.x < 2) {
        smem[threadIdx.x] = static_cast<V>(0);
    }
    __syncthreads();

    /*!
     * \note    we conduct round up here to make sure threads participate
     *          in the loop, as all threads will need the final_value with
     *          the return of this function
     */
    size_t numIterations = cudamop::utils::roundUp<size_t>(n, static_cast<size_t>(blockDim.x));

    for(i=threadIdx.x; i<numIterations; i++){
        bool inRange = (i < n);
        V value = inRange ? (data[i]) : static_cast<V>(0);
        
        // this should not be conflict as the top-kth value is unique
        if(inRange && ((TypeTranslator<V>::convert(value) & desiredMask) == desired)){
            smem[0] = static_cast<V>(1);
            smem[1] = value;
        }

        __syncthreads();

        V found = smem[0];
        V final_value = smem[1];

        __syncthreads();

        if(found != static_cast<V>(0)){
            return final_value;
        }
    }

    // shouldn't reach here
    assert(false);
    return static_cast<V>(0);
}


/*!
 * \brief   returns the top-kth (bottom-kth) element found in the data using radix selection
 * \tparam  V           data type
 * \tparam  isTopK      true for top-kth, false for bottom-kth
 * \param   data        input data
 * \param   k           top-kth value
 * \param   n           size of input data
 * \param   smem        shared memory area
 * \param   result      the selected final top-kth (bottom-kth) value
 */
template<typename V, bool isTopK>
__device__ void radixSelectTopKth(V* data, size_t k, size_t n, unsigned int* smem, V* result) {
    // per-thread buckets into which we accumulate digit counts in our radix
    unsigned int counts[RADIX_SIZE];
    
    RadixType desired = 0;
    RadixType desiredMask = 0;

    size_t kToFind = k;    
    int digitPos;
    int i;

    for(digitPos = sizeof(V)*8-RADIX_BITS; digitPos >= 0; digitPos -= RADIX_BITS){
        // count radix distribution for the current position
        countRadixUsingMask<V, RADIX_SIZE, RADIX_BITS>(data, n, counts, smem, digitPos, desired, desiredMask);

        /*!
         * \brief   to check whether reach the following condition:
         *              [1] count == 1: found the exact one value;
         *              [2] kToFind == 1: remains only one element to find;
         *          the above condition implies that we have found the unique top-kth value's pattern,
         *          yet the value itself might still be incomplete
         * \param   radixDigits the current radix digits we are processing
         * \param   count       number of the current radix digits within the data
         * \return  true if we find the top-kth value; vice versa is false
         */
        auto found_unique = [&](RadixType radixDigits, size_t count) -> bool {
            if(count == 1 && kToFind == 1){
                desired 
                    = cudamop::utils::BitField<RadixType>::setBitField(desired, radixDigits, digitPos, RADIX_BITS);
                desiredMask 
                    = cudamop::utils::BitField<RadixType>::setBitField(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

                /*!
                 * \note    currently we have found the one and only top-kth value's pattern, yet the
                 *          value itself might be still incomplete, hence we need to find out this 
                 *          value with the incompleted desired digits.
                 */
                *result = findPattern<V>((V*)smem, data, n, desired, desiredMask);
                return true;
            } else {
                return false;
            }
        };

        /*!
         * \brief   forward radix digit mask
         * \param   radixDigits the current radix digits we are processing
         * \param   count       number of the current radix digits within the data
         * \return  true if we have finishing forwarding; false for vice versa
         */
        auto found_non_unique = [&](RadixType radixDigits, size_t count) -> bool {
            if(count >= kToFind){
                desired 
                    = cudamop::utils::BitField<RadixType>::setBitField(desired, radixDigits, digitPos, RADIX_BITS);
                desiredMask 
                    = cudamop::utils::BitField<RadixType>::setBitField(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
                
                /*!
                 *  \note   the top-Kth element v must now be one such that:
                 *          (v & desiredMask == desired)
                 *          but we haven't narrowed it down; we must check the next
                 *          least-significant digit
                 */
                return true;
            } else {
                /*! 
                 * \note    {count} of values that hold {radixDigits} is larger than the value we are finding
                 *          to exclude them so kToFind will decrease {count} 
                 */
                kToFind -= count;
                return false;
            }
        };

        /*!
         *  \note   in the following code, all threads within the block will 
         *          doing the exact same thing to find out the kth value
         */
        if(isTopK){
#pragma unroll
            for(i=RADIX_SIZE-1; i>=0; i--){
                unsigned int count = counts[i];
                if(found_unique(i, count)){ return; }
                if(found_non_unique(i, count)){ break; }
            }
        } else {
#pragma unroll
            for(i=0; i<RADIX_SIZE; i++){
                unsigned int count = counts[i];
                if(found_unique(i, count)){ return; }
                if(found_non_unique(i, count)){ break; }
            }
        }

    } // for(digitPos=sizeof(V)*8-RADIX_BITS; digitPos>=0; digitPos-=RADIX_BITS)

    // translate the value from RadixType to V, and write out the final value
    *result = TypeTranslator<V>::deconvert(desired);
}

} // namespace radix

} // namespace select

} // namespace cudamop

#endif