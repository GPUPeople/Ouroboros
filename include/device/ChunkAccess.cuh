#pragma once

#include "Parameters.h"
#include "Definitions.h"
#include "Utility.cuh"

template <size_t SIZE, size_t SMALLEST_PAGE>
struct ChunkAccess
{
	static constexpr size_t Chunk_Size_{SIZE};
	static constexpr size_t smallest_page_size_{SMALLEST_PAGE};
	static constexpr unsigned int MaximumBitMaskSize_{static_cast<unsigned int>(Chunk_Size_ / (smallest_page_size_ * sizeof(uint64_t) * BYTE_SIZE))};

	enum class Mode
	{
		ERROR, SUCCESSFULL, DEQUEUE_CHUNK, RE_ENQUEUE_CHUNK, CONTINUE
	};

	enum class FreeMode
	{
		SUCCESS, FIRST_FREE, DEQUEUE
	};

	// Members
	int count; // Number of available pages
	uint32_t size; // Number of pages
	unsigned long long availability_mask[MaximumBitMaskSize_]; // Bitmask of how many are free

	// Methods
	__forceinline__ __device__ ChunkAccess(int available_pages, uint32_t number_pages) : count{ available_pages }, size{ number_pages }
	{
		// Setup Availability mask
		auto taken_pages = number_pages - available_pages;
		#pragma unroll
		for (auto i = 0; i < MaximumBitMaskSize_; ++i)
		{
			if(taken_pages >= Ouro::sizeofInBits<uint64_t>()) 
			{
				availability_mask[i] = 0ULL;
				taken_pages -= Ouro::sizeofInBits<uint64_t>();
			}
			else
			{
				// Set the rest of the bits (set the next greater bit for taken pages, -1 gives bitmask with 1 for taken pages, invert that
				if(number_pages >= Ouro::sizeofInBits<uint64_t>())
				{
					availability_mask[i] = ~((1ULL << (taken_pages)) - 1);
				}
				else
				{
					// Mask out top bits
					availability_mask[i] = ~((1ULL << (taken_pages)) - 1) & ((1ULL << number_pages) - 1);
				}
				taken_pages = 0;
			}

			if(number_pages > Ouro::sizeofInBits<uint64_t>())
			{
				number_pages -= Ouro::sizeofInBits<uint64_t>();
			}
			else
			{
				number_pages = 0;
			}
		}
	}

	__forceinline__ __device__ ChunkAccess(uint32_t number_pages) : count{ static_cast<int>(number_pages) }, size{ number_pages }
	{
		// Setup Availability mask
		#pragma unroll
		for (auto i = 0; i < MaximumBitMaskSize_; ++i)
		{
			// Set the rest of the bits (set the next greater bit for taken pages, -1 gives bitmask with 1 for taken pages, invert that
			if(number_pages >= Ouro::sizeofInBits<uint64_t>())
			{
				// All bits set
				availability_mask[i] = ~0ULL;
			}
			else
			{
				// Mask out top bits
				availability_mask[i] = (1ULL << number_pages) - 1;
			}

			number_pages = (number_pages > Ouro::sizeofInBits<uint64_t>()) ? (number_pages - Ouro::sizeofInBits<uint64_t>()) : 0;
		}
	}

	__forceinline__ __device__ unsigned long long createBitPattern(int bit)
	{
		return ~(1ULL << bit);
	}

	__forceinline__ __device__ bool checkBitSet(unsigned long long value, int bit)
	{
		return (1ULL << bit) & value;
	}

	// Free up a page, manipulating the count and availability mask, locking required
	__forceinline__ __device__ FreeMode freePage(index_t queue_index);

	// Try to allocate a page, locking required
	__forceinline__ __device__ Mode allocPage(index_t& page_index);

	__forceinline__ __device__ bool tryFlashChunk();
};