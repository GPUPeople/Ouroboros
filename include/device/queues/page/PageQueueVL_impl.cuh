#pragma once
#include "Parameters.h"
#include "PageQueueVL.cuh"
#include "device/BulkSemaphore_impl.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryIndex.cuh"
#include "device/queues/QueueChunk_impl.cuh"

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void PageQueueVL<CHUNK_TYPE>::init(MemoryManagerType* memory_manager)
{
	if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
	{
		// Allocate 1 chunk per queue in the beginning
		index_t chunk_index{0};
		memory_manager->allocateChunk<true>(chunk_index);
		auto queue_chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, 0);

		if(!FINAL_RELEASE && printDebug)
			printf("Allocate a new chunk for the queue %u with index: %u : ptr: %p\n",queue_index_, chunk_index, queue_chunk);
		
		// All pointers point to the same chunk in the beginning
		front_ptr_ = queue_chunk;
		back_ptr_ = queue_chunk;
		old_ptr_ = queue_chunk;
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ bool PageQueueVL<CHUNK_TYPE>::enqueueChunk(MemoryManagerType* memory_manager, index_t chunk_index, index_t pages_per_chunk)
{
	unsigned int virtual_pos = atomicAdd(&back_, pages_per_chunk);

	back_ptr_->enqueueChunk(memory_manager, virtual_pos, chunk_index, pages_per_chunk, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);

	// Please DO NOT reorder here
	__threadfence_block();

	// Since our space is not limited, we can signal at the end
	semaphore.signalExpected(pages_per_chunk);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ bool PageQueueVL<CHUNK_TYPE>::enqueueInitialChunk(MemoryManagerType* memory_manager, index_t chunk_index, int available_pages, index_t pages_per_chunk)
{
	const auto start_page_index = pages_per_chunk - available_pages;

	unsigned int virtual_pos = atomicAdd(&back_, available_pages);
	back_ptr_->enqueueChunk(memory_manager, virtual_pos, chunk_index, available_pages, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_, start_page_index);

	semaphore.signal(available_pages);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void* PageQueueVL<CHUNK_TYPE>::allocPage(MemoryManagerType* memory_manager)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	MemoryIndex index;
	uint32_t chunk_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);

	semaphore.wait(1, pages_per_chunk, [&]()
	{
		if (!memory_manager->allocateChunk<false>(chunk_index))
		{
			if(!FINAL_RELEASE)
				printf("TODO: Could not allocate chunk!!!\n");
		}

	 	ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
		__threadfence();
	 	enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
	});

	__threadfence();

	// unsigned int virtual_pos = atomicAdd(&front_, 1);
	unsigned int virtual_pos = Ouro::atomicAggInc(&front_);
	front_ptr_->template dequeue<QueueChunkType::DEQUEUE_MODE::DEQUEUE>(memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);

	chunk_index = index.getChunkIndex();
	return ChunkType::getPage(memory_manager->d_data, chunk_index, index.getPageIndex(), page_size_);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void PageQueueVL<CHUNK_TYPE>::freePage(MemoryManagerType* memory_manager, MemoryIndex index)
{
	enqueue(memory_manager, index.index);
	
	__threadfence_block();
	
	semaphore.signal(1);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void PageQueueVL<CHUNK_TYPE>::enqueue(MemoryManagerType* memory_manager, index_t index)
{
	// Increase back and compute the position on a chunk
	// const unsigned int virtual_pos = atomicAdd(&back_, 1);
	unsigned int virtual_pos = Ouro::atomicAggInc(&back_);
	back_ptr_->enqueue(memory_manager, virtual_pos, index, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);
}