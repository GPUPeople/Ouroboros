#pragma once

#include "Utility.cuh"

struct BulkSemaphore
{
	// | --- Counter (20 + 1 bits) --- | --- Expected (21 bits) --- | --- Reserved (21 bits) --- |

public:
	
	// ################################################################################################################
	// Primitives
	__forceinline__ __device__ unsigned long long create64BitSubAdder_expected(unsigned long long N)
	{
		return ~(N << middle_mask_shift) + 1;
	}

	__forceinline__ __device__ unsigned long long create64BitSubAdder_reserved(unsigned long long N)
	{
		return ~(N << upper_mask_shift) + 1;
	}

	__forceinline__ __device__ void getValues(int& count, int& expected, int& reserved) const
	{
		count = getCount();
		expected = static_cast<int>((value >> middle_mask_shift) & (highest_value_mask));
		reserved = static_cast<int>((value >> upper_mask_shift) & (highest_value_mask));
	}

	// Create a new value
	static __forceinline__ __device__ unsigned long long createValueExternal(int count, int expected, int reserved)
	{
		return static_cast<unsigned long long>(count) + null_value
		+ (static_cast<unsigned long long>(expected) << middle_mask_shift)
		+ (static_cast<unsigned long long>(reserved) << upper_mask_shift);
	}

	// Create a new value
	__forceinline__ __device__ void createValueInternal(int count, int expected, int reserved)
	{
		value = createValueExternal(count, expected, reserved);
	}

	__forceinline__ __device__ void read(BulkSemaphore& semaphore)
	{
		semaphore.value = Ouro::ldg_cg(&value);
	}

	// ################################################################################################################
	// Static Variables
	static constexpr unsigned long long middle_mask_shift{ 25 };
	static constexpr unsigned long long upper_mask_shift{ middle_mask_shift * 2 };
	// Highest expressable value
	static constexpr unsigned long long highest_value_mask{ (1ULL << middle_mask_shift) - 1 };
	// Additive Subtraction value for 1
	static constexpr unsigned long long subtract_one_value{~(1ULL) + 1};

public:

	__device__ BulkSemaphore() : value{null_value} {}
	__device__ BulkSemaphore(unsigned long long init_value) : value{init_value}{}
	
	// Value extractor
	__forceinline__ __device__ __host__ int getCount() const
	{
		return static_cast<int>(value & highest_value_mask) - null_value;
	}

	// Try to allocate some resource
	template <typename T>
	__forceinline__ __device__ void wait(int N, uint32_t number_pages_on_chunk, T allocationFunction);

	// Try to increase resources
	__forceinline__ __device__ bool tryReduce(int N);

	// Free a resource (number of pages only set for the thread which allocated stuff)
	__forceinline__ __device__ int signalExpected(unsigned long long N);

	// Free a resource (number of pages only set for the thread which allocated stuff)
	__forceinline__ __device__ int signal(unsigned long long N);

	unsigned long long value{null_value};
	static constexpr unsigned long long null_value {(1ULL << (middle_mask_shift - 1))};
};