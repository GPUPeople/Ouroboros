#pragma once

#include "device/Chunk.cuh"

template <typename ChunkBase, size_t SIZE>
struct PageChunk : public CommonChunk
{
	using Base = ChunkBase;
	static constexpr size_t size_{SIZE};
	static constexpr size_t meta_data_size_{CHUNK_METADATA_SIZE};

	// Members
	uint32_t number_pages;

	// ##########################################################################################################################
	// ##########################################################################################################################
	// Methods
	// ##########################################################################################################################
	// ##########################################################################################################################
	__device__ PageChunk(const unsigned int page_size, uint32_t number_pages) : CommonChunk(page_size), number_pages{number_pages} {}

	__forceinline__ __device__ __host__ void* getPage(memory_t* memory, index_t chunk_index, uint32_t page_index)
	{
		return reinterpret_cast<void*>(reinterpret_cast<memory_t*>(Base::getData(memory, chunk_index)) + (page_index * page_size));
	}

	// ##########################################################################################################################
	// ##########################################################################################################################
	// STATIC Methods
	// ##########################################################################################################################
	// ##########################################################################################################################
	
	static constexpr __forceinline__ __device__ __host__ size_t size() {return meta_data_size_ + size_;}

	static __forceinline__ __device__ __host__ void* getData(memory_t* memory, const index_t chunk_index) 
	{
		return Base::getData(memory, chunk_index);
	}

	static __forceinline__ __device__ __host__ void* getPage(memory_t* memory, const index_t chunk_index, const uint32_t page_index, const unsigned int page_size)
	{
		return Base::getPage(memory, chunk_index, page_index, page_size);
	}

	// template <typename QI>
	// static __forceinline__ __device__ index_t getQueueIndexFromPage(memory_t* memory, void* page)
	// {
	// 	const auto lowest_address = reinterpret_cast<unsigned long long>(memory);
	// 	const auto page_address = reinterpret_cast<unsigned long long>(page);
	// 	// Rounds down to the inverted chunk_index
	// 	const auto inverted_chunk_index = (page_address - lowest_address) / size();
	// 	// Take lowest address and add the inverted number of chunks to it
	// 	auto chunk = reinterpret_cast<PageChunk*>(lowest_address + (inverted_chunk_index * size()));

	// 	return QI::getQueueIndex(chunk->page_size);
	// }

	template <typename QI>
	static __forceinline__ __device__ index_t getQueueIndexFromPage(memory_t* memory, index_t chunk_index) 
	{
		auto chunk = reinterpret_cast<PageChunk*>(Base::getMemoryAccess(memory, chunk_index));
		return QI::getQueueIndex(chunk->page_size);
	}

	// ##############################################################################################################################################
	// 
	static __forceinline__ __device__ __host__ PageChunk* getAccess(memory_t* memory, index_t chunk_index)
	{
		return Base::template getMemoryAccess<PageChunk>(memory, chunk_index);
	}

	// ##############################################################################################################################################
	// Initializer
	static __forceinline__ __device__ __host__ PageChunk* initializeChunk(memory_t* memory, index_t chunk_index, uint32_t number_pages)
	{
		static_assert(Ouro::alignment(sizeof(PageChunk)) <= meta_data_size_, "PageChunk is larger than alignment!");
		return new(reinterpret_cast<char*>(getAccess(memory, chunk_index))) PageChunk((size_ / number_pages), number_pages);
	}

	// ##############################################################################################################################################
	// Initializer
	static __forceinline__ __device__ __host__ PageChunk* initializeChunk(memory_t* memory, index_t chunk_index, const int available_pages, uint32_t number_pages)
	{
		static_assert(Ouro::alignment(sizeof(PageChunk)) <= meta_data_size_, "PageChunk is larger than alignment!");
		return new(Base::template getMemoryAccess<memory_t>(memory, chunk_index)) PageChunk((size_ / number_pages), number_pages);
	}
};