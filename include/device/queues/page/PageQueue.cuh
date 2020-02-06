#pragma once

#include "Definitions.h"
#include "device/BulkSemaphore.cuh"
#include "Parameters.h"
#include "device/PageChunk.cuh"

// Forward declaration
struct MemoryIndex;

template <typename CHUNK_TYPE>
struct PageQueue
{
	using SemaphoreType = BulkSemaphore;
	using ChunkType = CHUNK_TYPE;

	// Members
	index_t* queue_;
	SemaphoreType semaphore{ SemaphoreType::null_value };
	unsigned int front_{ 0 };
	unsigned int back_{ 0 };
	int queue_index_{ 0 };
	int page_size_{ 0 };

	// Static Members
	static constexpr bool virtualized{false};
	static constexpr int size_{ page_queue_size };
	static_assert(isPowerOfTwo(size_), "Page Queue size is not Power of 2!");
	static constexpr int lower_fill_level{static_cast<int>(static_cast<float>(size_) * LOWER_FILL_LEVEL_PERCENTAGE)};

	// Methods
	__forceinline__ __device__ bool enqueue(index_t chunk_index);

	__forceinline__ __device__ bool enqueueChunk(index_t chunk_index, index_t pages_per_chunk);

	__forceinline__ __device__ void dequeue(MemoryIndex& index);

	template <typename MemoryManagerType>
	__forceinline__ __device__ void init(MemoryManagerType* memory_manager);

	template <typename MemoryManagerType>
	__forceinline__ __device__ bool preFillQueue(MemoryManagerType* memory_manager, index_t chunk_index, index_t pages_per_chunk)
	{
		return enqueueChunk(chunk_index, pages_per_chunk);
	}

	template <typename MemoryManagerType>
	__forceinline__ __device__ bool enqueueInitialChunk(MemoryManagerType* memory_manager, index_t chunk_index, int available_pages, index_t pages_per_chunk);

	template <typename MemoryManagerType>
	__forceinline__ __device__ void* allocPage(MemoryManagerType* memory_manager);

	template <typename MemoryManagerType>
	__forceinline__ __device__ void freePage(MemoryManagerType* memory_manager, MemoryIndex index);

	void resetQueue()
	{
		semaphore = SemaphoreType::null_value;
		front_ = 0;
		back_ = 0;
	}

	__forceinline__ __device__ __host__ uint32_t getCount()
	{
		return semaphore.getCount();
	}
};