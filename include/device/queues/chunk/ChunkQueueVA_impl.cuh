#pragma once
#include "Parameters.h"
#include "ChunkQueueVA.cuh"
#include "device/ChunkAccess_impl.cuh"
#include "device/BulkSemaphore_impl.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryIndex.cuh"
#include "device/queues/QueueChunk_impl.cuh"

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void ChunkQueueVA<CHUNK_TYPE>::init(MemoryManagerType* memory_manager)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
	{
		queue_[i] = DeletionMarker<index_t>::val;
	}

	if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
	{
		// Allocate 1 chunk per queue in the beginning
		index_t chunk_index{0};
		memory_manager->allocateChunk<true>(chunk_index);
		auto chunk = QueueChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, chunk_index, 0);
		queue_[0] = chunk_index;
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ bool ChunkQueueVA<CHUNK_TYPE>::enqueueChunk(MemoryManagerType* memory_manager, index_t chunk_index, index_t pages_per_chunk)
{
	if (atomicAdd(&count_, 1) < static_cast<int>(num_spots_))
	{
		enqueue(memory_manager, chunk_index);
		// Please do NOT reorder here
		__threadfence_block();
		semaphore.signalExpected(pages_per_chunk);
		return true;
	}

	if (!FINAL_RELEASE)
		printf("Queue %d: We died in EnqueueChunk with count %d\n", queue_index_, count_);
	__trap(); //no space to enqueue -> fail
	return false;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ bool ChunkQueueVA<CHUNK_TYPE>::enqueueInitialChunk(MemoryManagerType* memory_manager, 
	index_t chunk_index, int available_pages, index_t pages_per_chunk)
{
	// Increase count, insert chunk into queue
	++count_;
	++back_;
	auto queue_chunk = accessQueueElement(memory_manager, 0, 0);
	queue_chunk->enqueueInitial(0, chunk_index);
	semaphore.signal(available_pages);

	// Allocate one additional queue chunk
	index_t new_chunk_index{ 0 };
	memory_manager->allocateChunk<true>(new_chunk_index);
	QueueChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, new_chunk_index, QueueChunkType::num_spots_);
	queue_[1] = new_chunk_index;
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void* ChunkQueueVA<CHUNK_TYPE>::allocPage(MemoryManagerType* memory_manager)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	uint32_t page_index, chunk_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
	ChunkType* chunk{ nullptr };

	semaphore.wait(1, pages_per_chunk, [&]()
	{
		if (!memory_manager->allocateChunk<false>(chunk_index))
			printf("TODO: Could not allocate chunk!!!\n");

		ChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, chunk_index, pages_per_chunk, pages_per_chunk);
		// Please do NOT reorder here
		__threadfence_block();
		enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
	});

	unsigned int virtual_pos = Ouro::ldg_cg(&front_);
	while (true)
	{
		auto chunk_id = computeChunkID(virtual_pos);
		auto queue_chunk = accessQueueElement(memory_manager, chunk_id, virtual_pos);
		queue_chunk->access(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos), chunk_index);
		if (chunk_index != DeletionMarker<index_t>::val)
		{
			chunk = ChunkType::getAccess(memory_manager->d_data, memory_manager->start_index, chunk_index);
			const auto mode = chunk->access.allocPage(page_index);
			if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
				break;
			if (mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
			{
				// Pretty special case, but we simply enqueue in the end again
				if (atomicAdd(&count_, 1) < static_cast<int>(num_spots_))
				{
					enqueue(memory_manager, chunk_index);
				}
				break;
			}
			if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
			{
				// TODO: Why does this not work
				// atomicMax(&front_, virtual_pos + 1);
				// // Reduce count again
				// atomicSub(&count_, 1);

				// // We moved the front pointer
				// if(queue_chunk->deleteElement(virtual_pos % QueueChunkType::num_spots_))
				// {
				// 	// We can remove this chunk
				// 	index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
				// 	if(printDebug)
				// 		printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | AllocPage-Reuse\n", reusable_chunk_id, chunk_id, queue_chunk->virtual_start_);
				// 	//memory_manager->d_chunk_reuse_queue.enqueue(reusable_chunk_id);
				// }
				if (atomicCAS(&front_, virtual_pos, virtual_pos + 1) == virtual_pos)
				{
					//printf("We can dequeue this chunk %u\n", chunk_id);
					// Reduce count again
					atomicSub(&count_, 1);

					// We moved the front pointer
					if(queue_chunk->deleteElement(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos)))
					{
						// We can remove this chunk
						index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
						if(printDebug)
							printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | AllocPage-Reuse\n", reusable_chunk_id, chunk_id, queue_chunk->virtual_start_);
						memory_manager->d_chunk_reuse_queue.enqueue(reusable_chunk_id);
					}
				}
				break;
			}
		}

		// Check next chunk
		++virtual_pos;
		// ##############################################################################################################
		// Error Checking
		if (!FINAL_RELEASE)
		{
			if (virtual_pos > Ouro::ldg_cg(&back_))
			{
				if (!FINAL_RELEASE)
					printf("ThreadIDx: %d BlockIdx: %d - We done fucked up! Front: %u Back: %u : Count: %d\n", threadIdx.x, blockIdx.x, virtual_pos, back_, count_);
				__trap();
			}
		}
	}

	return ChunkType::getPage(memory_manager->d_data, memory_manager->start_index, chunk_index, page_index, page_size_);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ void ChunkQueueVA<CHUNK_TYPE>::freePage(MemoryManagerType* memory_manager, MemoryIndex index)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	auto chunk = ChunkType::getAccess(memory_manager->d_data, memory_manager->start_index, index.getChunkIndex());
	auto mode = chunk->access.freePage(index.getPageIndex());
	if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
	{
		// Please do NOT reorder here
		__threadfence_block();

		// We are the first to free something in this chunk, add it back to the queue
		if (atomicAdd(&count_, 1) < static_cast<int>(num_spots_))
		{
			enqueue(memory_manager, index.getChunkIndex());
		}
		else
		{
			if (!FINAL_RELEASE)
				printf("Queue %d: We died in FreePage with count %d\n", queue_index_, count_);
			__trap();
		}
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
__forceinline__ __device__ void ChunkQueueVA<CHUNK_TYPE>::enqueue(MemoryManagerType* memory_manager, index_t index)
{
	const unsigned int virtual_pos = atomicAdd(&back_, 1);
	auto chunk_id = computeChunkID(virtual_pos);
	const auto position = (virtual_pos % QueueChunkType::num_spots_);

	if (position == 0)
	{
		unsigned int chunk_index{ 0 };
		// We pre-emptively allocate the next chunk already
		memory_manager->allocateChunk<true>(chunk_index);
		QueueChunkType::initializeChunk(memory_manager->d_data, memory_manager->start_index, chunk_index, virtual_pos + QueueChunkType::num_spots_);

		__threadfence_block();

		atomicExch(&queue_[(chunk_id + 1) % size_], chunk_index);
	}

	__threadfence_block();

	auto chunk = accessQueueElement(memory_manager, chunk_id, virtual_pos);
	if(QueueChunkType::checkChunkEmptyEnqueue(chunk->enqueue(position, index)))
	{
		// We can remove this chunk
		index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
	  		Ouro::sleep();
		if(printDebug)
			printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | ENQUEUE-Reuse\n", reusable_chunk_id, chunk_id, chunk->virtual_start_);
		memory_manager->d_chunk_reuse_queue.enqueue(reusable_chunk_id);
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__forceinline__ __device__ QueueChunk<typename CHUNK_TYPE::Base>* ChunkQueueVA<CHUNK_TYPE>::accessQueueElement(MemoryManagerType* memory_manager, index_t chunk_id, index_t v_position)
{
	index_t queue_chunk_index{0};
	// We may have to wait until the first thread on this chunk has initialized it!
	while((queue_chunk_index = Ouro::ldg_cg(&queue_[chunk_id])) == DeletionMarker<index_t>::val) 
	{
		Ouro::sleep();
	}

	__threadfence_block();

	auto queue_chunk = QueueChunkType::getAccess(memory_manager->d_data, memory_manager->start_index, queue_chunk_index);
	if(!queue_chunk->checkVirtualStart(v_position))
	{
		if (!FINAL_RELEASE)
			printf("Virtualized does not match for chunk: %u at position: %u with virtual start: %u  ||| v_pos: %u\n", queue_chunk_index, chunk_id, queue_chunk->virtual_start_, v_position);
		__trap();
	}

	return queue_chunk;
}