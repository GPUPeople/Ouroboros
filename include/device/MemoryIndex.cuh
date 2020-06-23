#pragma once

#include "Parameters.h"
#include "Definitions.h"

struct MemoryIndex
{
	static constexpr unsigned int NumBitsForPage{NUM_BITS_FOR_PAGE};
	static constexpr unsigned int NumBitsForChunk{ 32 - NumBitsForPage };
	static constexpr unsigned int MaxNumChunks{ 1 << NumBitsForChunk };
	static constexpr unsigned int PageBitMask{(1 << NUM_BITS_FOR_PAGE) - 1};
	static constexpr uint32_t MAX_VALUE{0xFFFFFFFF};

	// Data	
	uint32_t index;

	__device__ MemoryIndex() : index{0U}{}
	__device__ MemoryIndex(uint32_t chunk_index, uint32_t page_index) : index{(chunk_index << NumBitsForPage) + page_index}{}

	// Methods
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ uint32_t getIndex() { return index; }
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ void getIndex(uint32_t& chunk_index, uint32_t& page_index)
	{
		const auto temp_index = index;
		chunk_index = temp_index >> NumBitsForPage;
		page_index = temp_index & PageBitMask;
	}
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ uint32_t getChunkIndex()
	{
		return index >> NumBitsForPage;
	}
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ uint32_t getPageIndex()
	{
		return index & PageBitMask;
	}
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ static constexpr uint32_t createIndex(uint32_t chunk_index, uint32_t page_index)
	{
		return (chunk_index << NumBitsForPage) + page_index;
	}
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ void setIndex(uint32_t ind) { index = ind; }
	// ----------------------------------------------------------------------------
	__device__ __forceinline__ void setIndex(uint32_t chunk_index, uint32_t page_index)
	{
		index = createIndex(chunk_index, page_index);
	}
};