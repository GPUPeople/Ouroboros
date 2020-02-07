#pragma once
#include "Parameters.h"
#include "ChunkQueueVL.cuh"
#include "device/ChunkAccess_impl.cuh"
#include "device/BulkSemaphore_impl.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryIndex.cuh"
#include "device/queues/QueueChunk_impl.cuh"

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void ChunkQueueVL<CHUNK_TYPE>::init(MemoryManagerType* memory_manager)
{
	if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
	{
		// Allocate 1 chunk per queue in the beginning
		index_t chunk_index{0};
		memory_manager->allocateChunk(chunk_index);
		auto queue_chunk = QueueChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, chunk_index, 0);

		if(printDebug)
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
__forceinline__ __device__ bool ChunkQueueVL<CHUNK_TYPE>::enqueueChunk(MemoryManagerType* memory_manager, index_t chunk_index, index_t pages_per_chunk)
{
	enqueue(memory_manager, chunk_index);

	// Please do NOT reorder here
	__threadfence();

	semaphore.signalExpected(pages_per_chunk);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ bool ChunkQueueVL<CHUNK_TYPE>::enqueueInitialChunk(MemoryManagerType* memory_manager, 
	index_t chunk_index, int available_pages, index_t pages_per_chunk)
{
	enqueue(memory_manager, chunk_index);
	semaphore.signal(available_pages);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void* ChunkQueueVL<CHUNK_TYPE>::allocPage(MemoryManagerType* memory_manager)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	uint32_t chunk_index, page_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
	ChunkType* chunk{ nullptr };

	semaphore.wait(1, pages_per_chunk, [&]()
	{
		if (!memory_manager->allocateChunk(chunk_index))
			printf("TODO: Could not allocate chunk!!!\n");

		ChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, chunk_index, pages_per_chunk, pages_per_chunk);
		__threadfence();
		enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
		__threadfence();
	});

	__threadfence_block();

	unsigned int virtual_pos = ldg_cg(&front_);
	while(true)
	{
		front_ptr_->accessLinked(virtual_pos, chunk_index);

		__threadfence_block();

		// This position might be out-dated already
		if(chunk_index != DeletionMarker<index_t>::val)
		{
			chunk = ChunkType::getAccess(memory_manager->d_data, memory_manager->start_index, chunk_index);
			const auto mode = chunk->access.allocPage(page_index);
			
			if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
				break;
			if (mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
			{
				// Pretty special case, but we simply enqueue in the end again
				enqueue(memory_manager, chunk_index);
				break;
			}
			if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
			{
				if (atomicCAS(&front_, virtual_pos, virtual_pos + 1) == virtual_pos)
				{
					front_ptr_->dequeue<QueueChunkType::DEQUEUE_MODE::DELETE>(memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
				}
				// TODO: Why does this not work
				// atomicMax(&front_, virtual_pos + 1);
				// front_ptr_->dequeue<QueueChunkType::DEQUEUE_MODE::DELETE>(memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
				break;
			}
		}

		// Check next chunk
		++virtual_pos;
		// ##############################################################################################################
		// Error Checking
		if (!FINAL_RELEASE)
		{
			if (virtual_pos > ldg_cg(&back_))
			{
				if (!FINAL_RELEASE)
					printf("ThreadIDx: %d BlockIdx: %d - Front: %u Back: %u - ChunkIndex: %u\n", threadIdx.x, blockIdx.x, virtual_pos, back_, chunk_index);
				__trap();
			}
		}
	}

	return ChunkType::getPage(memory_manager->d_data, memory_manager->start_index, chunk_index, page_index);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void ChunkQueueVL<CHUNK_TYPE>::freePage(MemoryManagerType* memory_manager, MemoryIndex& index)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	uint32_t chunk_index, page_index;
	index.getIndex(chunk_index, page_index);
	auto chunk = ChunkType::getAccess(memory_manager->d_data, memory_manager->start_index, chunk_index);
	auto mode = chunk->access.freePage(page_index);
	if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
	{
		enqueue(memory_manager, chunk_index);
	}
	else if(mode == ChunkType::ChunkAccessType::FreeMode::DEQUEUE)
	{
		// TODO: Implement dequeue chunks
		if(printDebug)
			printf("I guess I should actually dequeue at this point!\n");
	}
	// Please do NOT reorder here
	__threadfence_block();

	// Signal a free page
	semaphore.signal(1);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void ChunkQueueVL<CHUNK_TYPE>::enqueue(MemoryManagerType* memory_manager, index_t index)
{
	// Increase back and compute the position on a chunk
	const unsigned int virtual_pos = atomicAdd(&back_, 1);
	back_ptr_->enqueue(memory_manager, virtual_pos, index, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);
}