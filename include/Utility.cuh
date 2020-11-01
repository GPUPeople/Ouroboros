#pragma once
#include "Utility.h"

namespace Ouro
{
	__forceinline__ __device__ unsigned int ldg_cg(const unsigned int* src)
	{
		unsigned int dest{ 0 };
	#ifdef __CUDA_ARCH__
		asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	#endif
		return dest;
	}

	__forceinline__ __device__ int ldg_cg(const int* src)
		{
			int dest{ 0 };
	#ifdef __CUDA_ARCH__
			asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	#endif
			return dest;
		}

	__forceinline__ __device__ unsigned long long ldg_cg(const unsigned long long* src)
	{
		unsigned long long dest{0};
	#ifdef __CUDA_ARCH__
		asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	#endif
		return dest;
	}

	__forceinline__ __device__ const unsigned int& stg_cg(unsigned int* dest, const unsigned int& src)
	{
	#ifdef __CUDA_ARCH__
		asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(dest), "r"(src));
	#endif
		return src;
	}

	__forceinline__ __device__ void store(volatile uint4* dest, const uint4& src)
	{
	#ifdef __CUDA_ARCH__
		asm("st.volatile.v4.u32 [%0], {%1, %2, %3, %4};"
			:
		: "l"(dest), "r"(src.x), "r"(src.y), "r"(src.z), "r"(src.w));
	#endif
	}

	__forceinline__ __device__ void store(volatile uint2* dest, const uint2& src)
	{
	#ifdef __CUDA_ARCH__
		asm("st.volatile.v2.u32 [%0], {%1, %2};"
			:
		: "l"(dest), "r"(src.x), "r"(src.y));
	#endif
	}

	static __forceinline__ __device__ int lane_id()
	{
		return threadIdx.x & (WARP_SIZE - 1);
	}

	__forceinline__ __device__ void sleep(unsigned int factor = 1)
	{
	#ifdef __CUDA_ARCH__
	#if (__CUDA_ARCH__ >= 700)
		//__nanosleep(SLEEP_TIME);
		__nanosleep(SLEEP_TIME * factor);
	#else
		__threadfence();
	#endif
	#endif
	}

	__forceinline__ __device__ int atomicAggInc(unsigned int *ptr)
	{
		#ifdef __CUDA_ARCH__
		#if(__CUDA_ARCH__ >= 700)
		int mask = __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(ptr));
		int leader = __ffs(mask) - 1;
		int res;
		if(lane_id() == leader)
			res = atomicAdd(ptr, __popc(mask));
		res = __shfl_sync(mask, res, leader);
		return res + __popc(mask & ((1 << lane_id()) - 1));
		#else
		return atomicAdd(ptr, 1);
		#endif
		#else
		auto val = *ptr;
		*ptr += 1;
		return val;
		#endif
	}
}

