#pragma once
#include "Parameters.h"
#include "device/queues/page/PageQueue.cuh"
#include "device/MemoryIndex.cuh"
#include "device/BulkSemaphore_impl.cuh"

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__forceinline__ __device__ void PageQueue<ChunkType>::init(MemoryManagerType* memory_manager)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
	{
		queue_[i] = DeletionMarker<index_t>::val;
	}
}

// ##############################################################################################################################################
//
template <typename ChunkType>
__forceinline__ __device__ bool PageQueue<ChunkType>::enqueue(index_t chunk_index)
{
	if (semaphore.signal(1) < size_)
	{
		//we have to wait in case there is still something in the spot
		// note: as the filllevel could be increased by this thread, we are certain that the spot will become available
		// unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, 1));
		unsigned int pos = Ouro::modPower2<size_>(Ouro::atomicAggInc(&back_));
		while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, chunk_index) != DeletionMarker<index_t>::val)
			Ouro::sleep();
		return true;
	}
	__trap(); //no space to enqueue -> fail
	return false;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__forceinline__ __device__ bool PageQueue<ChunkType>::enqueueInitialChunk(MemoryManagerType* memory_manager, index_t chunk_index, int availablePages, index_t pages_per_chunk)
{
	const auto start_page_index = pages_per_chunk - availablePages;

	for(auto i = start_page_index; i < pages_per_chunk; ++i)
	{
		queue_[back_++] = MemoryIndex::createIndex(chunk_index, i);
	}

	semaphore.signal(availablePages);
	return true;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
__forceinline__ __device__ bool PageQueue<ChunkType>::enqueueChunk(index_t chunk_index, index_t pages_per_chunk)
{
	if (semaphore.signalExpected(pages_per_chunk) < size_)
	{
		//we have to wait in case there is still something in the spot
		// note: as the filllevel could be increased by this thread, we are certain that the spot will become available
		unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, pages_per_chunk));
		for(auto i = 0; i < pages_per_chunk; ++i)
		{
			index_t index = MemoryIndex::createIndex(chunk_index, i);
			while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, index) != DeletionMarker<index_t>::val)
			{
				Ouro::sleep();
			}
				
			pos = Ouro::modPower2<size_>(++pos);
		}
		return true;
	}
	__trap(); //no space to enqueue -> fail
	return false;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
__forceinline__ __device__ void PageQueue<ChunkType>::dequeue(MemoryIndex& index)
{
	// Dequeue from queue
	// #if (__CUDA_ARCH__ < 700)
	// unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&front_, 1));
	// #else
	unsigned int pos = Ouro::modPower2<size_>(Ouro::atomicAggInc(&front_));
	// #endif
	auto counter {0U};
	while ((index.index = atomicExch(queue_ + pos, DeletionMarker<index_t>::val)) == DeletionMarker<index_t>::val)
	{
		Ouro::sleep(counter++);
		// if(counter++ > 1000000)
		// {
		// 	int count, reserved,expected;
		// 	semaphore.getValues(count, reserved, expected);
		// 	printf("%d - %d - %d We died in PageQueue::dequeue waiting on a value! :( pos: %u | %d - %d - %d\n", blockIdx.x, threadIdx.x, Ouro::lane_id(), pos, count, reserved, expected);
		// 	__trap();
		// }
	}
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__forceinline__ __device__ void* PageQueue<ChunkType>::allocPage(MemoryManagerType* memory_manager)
{
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
	 	enqueueChunk(chunk_index, pages_per_chunk);
	});
	
	// Get index from queue
	dequeue(index);

	// Return page to caller
	return ChunkType::getPage(memory_manager->d_data, index.getChunkIndex(), index.getPageIndex(), page_size_);
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__forceinline__ __device__ void PageQueue<ChunkType>::freePage(MemoryManagerType* memory_manager, MemoryIndex index)
{
	// Enqueue this index into the queue
	enqueue(index.index);
}