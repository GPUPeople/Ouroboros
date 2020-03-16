#pragma once

#include "Utility.cuh"
#include "device/MemoryIndex.cuh"
#include "device/Helper.cuh"
#include "Parameters.h"
#include "device/ChunkLocator.cuh"

struct CommonChunk
{
	__device__ CommonChunk(const unsigned int page_size) : page_size{page_size}{}
	unsigned int page_size{0U};
};

template <size_t ChunkSize>
struct Chunk
{
	static constexpr size_t size_{ChunkSize};
	static constexpr size_t meta_data_size_{CHUNK_METADATA_SIZE};

	// ###################################################################
	// Size
	static constexpr __forceinline__ __device__ __host__ size_t size()
	{
		return meta_data_size_ + size_;
	}

	// ##############################################################################################################################################
	//
	template <typename T = memory_t>
	static __forceinline__ __device__ __host__ T* getMemoryAccess(memory_t* memory, const index_t chunk_index)
	{
		return reinterpret_cast<T*>(&memory[chunk_index * size()]);
	}

	// ##############################################################################################################################################
	// 
	static __forceinline__ __device__ __host__ void* getData(memory_t* memory, const index_t chunk_index)
	{
		return getMemoryAccess<memory_t>(memory, chunk_index) + meta_data_size_;
	}

	// ##############################################################################################################################################
	// 
	static __forceinline__ __device__ __host__ void* getPage(memory_t* memory, const index_t chunk_index, const uint32_t page_index, const int page_size)
	{
		return reinterpret_cast<memory_t*>(getData(memory, chunk_index)) + (page_index * page_size);
	}

	// ##############################################################################################################################################
	// 
	static __forceinline__ __device__ __host__ index_t getIndexFromPointer(memory_t* memory, void* chunk)
	{
		// INFO: This will not always report the correct chunk index for MultiOuroboros, but this should not matter
		return (reinterpret_cast<unsigned long long>(chunk) - reinterpret_cast<unsigned long long>(memory)) / size();
	}

	// ##############################################################################################################################################
	// 
	template <unsigned int CHUNK_SIZE>
	static __forceinline__ __device__ MemoryIndex getPageIndexFromPointer(memory_t* memory, void* page, index_t page_size)
	{
		MemoryIndex index;
		// start index points to beginning of chunk 0, so go one above so the differences make sense
		const auto difference = reinterpret_cast<unsigned long long>(page) - reinterpret_cast<unsigned long long>(memory);
		// By dividing the left over difference by the size of a chunk, we should get the correct chunk index (due to rounding down)
		index_t chunk_index = (difference / size());
		// Subtract the alignments of all chunks before (so we are left with just the page data) and subtract one page to correct the index
		unsigned long long remaining_size = difference - (chunk_index * meta_data_size_) - page_size;
		// The remaining size mod the Chunksize should give us now the offset per chunk
		unsigned long long mod_per_chunk = remaining_size & (size_ - 1);
		// How many pages per chunk - our page - 1 should give us the correct index!
		index_t page_index = (size_ / page_size) - (mod_per_chunk / page_size) - 1;

		index.setIndex(chunk_index, page_index);
		return index;
	}
};