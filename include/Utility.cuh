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
}

