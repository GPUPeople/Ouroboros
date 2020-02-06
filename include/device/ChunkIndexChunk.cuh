#pragma once

#include "Chunk.cuh"

template <typename ChunkBase, size_t SIZE, size_t SMALLEST_PAGE>
struct ChunkIndexChunk : public CommonChunk
{
	static constexpr size_t size_{SIZE};
	static constexpr size_t meta_data_size_{CHUNK_METADATA_SIZE};
	static constexpr size_t smallest_page_size_{SMALLEST_PAGE};

	using Base = ChunkBase;
	using ChunkAccessType = ChunkAccess<size_, smallest_page_size_>;

	// Members
	ChunkAccessType access;
	unsigned int queue_pos{DeletionMarker<unsigned int>::val};
	unsigned int identifier{ CHUNK_IDENTIFIER };

	// ##########################################################################################################################
	// ##########################################################################################################################
	// Methods
	// ##########################################################################################################################
	// ##########################################################################################################################
	__device__ ChunkIndexChunk(const unsigned int page_size, const int available_pages, const uint32_t number_pages, const unsigned int queue_position) : 
		CommonChunk(page_size), access(available_pages, number_pages), queue_pos{queue_position} {}
	
	__device__ ChunkIndexChunk(const unsigned int page_size, const uint32_t number_pages, const unsigned int queue_position) : 
		CommonChunk(page_size), access(number_pages), queue_pos{queue_position} {}

	// ##########################################################################################################################
	// ##########################################################################################################################
	// Static Methods
	// ##########################################################################################################################
	// ##########################################################################################################################
	static constexpr __forceinline__ __device__ __host__ size_t size() {return meta_data_size_ + size_;}

	static __forceinline__ __device__ __host__ void* getData(memory_t* memory, const uint64_t start_index, const index_t chunk_index) 
	{
		return ChunkBase::getData(memory, start_index, chunk_index);
	}

	static __forceinline__ __device__ __host__ void* getPage(memory_t* memory, const uint64_t start_index, const index_t chunk_index, const uint32_t page_index)
	{
		return ChunkBase::getPage(memory, start_index, chunk_index, page_index, page_size);
	}

	static __forceinline__ __device__ __host__ ChunkIndexChunk* getAccess(memory_t* memory, const uint64_t start_index, const index_t chunk_index)
	{
		return reinterpret_cast<ChunkIndexChunk*>(Base::getMemoryAccess(memory, start_index, chunk_index));
	}

	template <typename QI>
	static __forceinline__ __device__ index_t getQueueIndexFromPage(memory_t* memory, const uint64_t start_index, void* page)
	{
		const auto lowest_address = reinterpret_cast<unsigned long long>(memory);
		const auto page_address = reinterpret_cast<unsigned long long>(page);
		// Rounds down to the inverted chunk_index
		const auto inverted_chunk_index = (page_address - lowest_address) / size();
		// Take lowest address and add the inverted number of chunks to it
		auto chunk = reinterpret_cast<ChunkIndexChunk*>(lowest_address + (inverted_chunk_index * size()));

		return QI::getQueueIndex(chunk->page_size);
	}

	template <typename QI>
	static __forceinline__ __device__ index_t getQueueIndexFromPage(memory_t* memory, const uint64_t start_index, index_t chunk_index) 
	{
		auto chunk = reinterpret_cast<ChunkIndexChunk*>(Base::getMemoryAccess(memory, start_index, chunk_index));
		return QI::getQueueIndex(chunk->page_size);
	}

	static __forceinline__ __device__ __host__ ChunkIndexChunk* initializeChunk(memory_t* memory, const uint64_t start_index, const index_t chunk_index, 
		const int available_pages, const uint32_t number_pages, const unsigned int queue_position = DeletionMarker<unsigned int>::val)
	{
		static_assert(alignment(sizeof(Chunk)) <= meta_data_size_, "Chunk is larger than alignment!");
		return new(reinterpret_cast<char*>(getAccess(memory, start_index, chunk_index))) ChunkIndexChunk((size_ / number_pages), available_pages, number_pages, queue_position);
	}

	static __forceinline__ __device__ __host__ ChunkIndexChunk* initializeEmptyChunk(memory_t* memory, const uint64_t start_index, const index_t chunk_index, 
		const uint32_t number_pages, const unsigned int queue_position = DeletionMarker<unsigned int>::val)
	{
		static_assert(alignment(sizeof(Chunk)) <= meta_data_size_, "Chunk is larger than alignment!");
		return new(reinterpret_cast<char*>(getAccess(memory, start_index, chunk_index))) ChunkIndexChunk((size_ / number_pages), number_pages, queue_position);
	}
};