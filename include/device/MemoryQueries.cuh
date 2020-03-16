#pragma once
#include "Parameters.h"
#include "Definitions.h"
#include "device/Ouroboros.cuh"

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::printFreeResources()
{
	__shared__ int num_pages_per_queue[NumberQueues_];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= 1)
		return;

	for(auto i = 0; i < NumberQueues_; ++i)
		num_pages_per_queue[i] = 0;

	__syncthreads();

	printf("-----------------------------------\nNumber of Chunks Overall: %6u\n-----------------------------------\n", *next_free_chunk - d_chunk_reuse_queue.count_);
	for(auto i = 0; i < *(next_free_chunk); ++i)
	{
		auto chunk = ChunkType::getAccess(d_data, i);
		const auto identifier = *reinterpret_cast<unsigned int*>(chunk);
		if(identifier == CHUNK_IDENTIFIER)
		{
			if (chunk->access.count)
			{
				printf("Chunk %6d - Pages in Chunk: %6u - Free Pages - %6u\n", i, chunk->access.size, chunk->access.count);
				num_pages_per_queue[QI::getQueueIndexFromNumPages(chunk->access.size)] += chunk->access.count;
			}
		}
	}
	printf("---------------------------------------------\n");
	for(auto i = 0; i < NumberQueues_; ++i)
	{
		if(num_pages_per_queue[i] != 0)
			printf("QueueIndex: %d - Free Pages: %d\n", i, num_pages_per_queue[i]);
	}
	printf("---------------------------------------------\n");
}

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::printFreeResources()
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= 1)
		return;

	printf("-----------------------------------\nQueue Fill Rates: \n-----------------------------------\n");
	for(auto i = 0; i < NumberQueues_; ++i)
	{
		auto count = d_storage_reuse_queue[i].getCount();
		if(count)
			printf("Queue %d: %u Pages free\n", i, count);
	}
}

template <typename OUROBOROS>
__global__ void d_printFreeResources(OUROBOROS* ouroboros)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid != 0)
		return;

	// Template-recursive give all memory managers the same pointer
	ouroboros->d_printResources();
}

template<class OUROBOROS, class... OUROBOROSES>
__forceinline__ __device__ void Ouroboros<OUROBOROS, OUROBOROSES...>::d_printResources()
{
	memory_manager.printFreeResources();
	next_memory_manager.d_printResources();
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
void Ouroboros<OUROBOROS, OUROBOROSES...>::printFreeResources()
{
	if(printDebug)
		d_printFreeResources<MyType> <<<1,1>>>(reinterpret_cast<MyType*>(memory.d_memory));
}