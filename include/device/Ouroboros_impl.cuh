#pragma once
#include "Ouroboros.cuh"
#include "device/queues/Queues_impl.cuh"
#include "device/MemoryQueries.cuh"
#include "device/queues/chunk/ChunkQueue_impl.cuh"

// ##############################################################################################################################################
//
//
// CHUNKS
//
//
// ##############################################################################################################################################

// ##############################################################################################################
// 
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void* OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::allocPage(size_t size)
{
	if(statistics_enabled)
		atomicAdd(&stats.pageAllocCount, 1);

	// Allocate from chunks
	return d_storage_reuse_queue[QI::getQueueIndex(size)].template allocPage<OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(this);

}

// ##############################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::freePage(MemoryIndex index)
{
	if(statistics_enabled)
		atomicAdd(&stats.pageFreeCount, 1);

	// Deallocate page in chunk
	d_storage_reuse_queue[QueueType::ChunkType::template getQueueIndexFromPage<QI>(d_data, start_index, index.getChunkIndex())].freePage(this, index);
}

// ##############################################################################################################################################
//
//
// PAGES
//
//
// ##############################################################################################################################################

// ##############################################################################################################
// 
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void* OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::allocPage(size_t size)
{
	if(statistics_enabled)
		atomicAdd(&stats.pageAllocCount, 1);

	// Allocate from pages
	return d_storage_reuse_queue[QI::getQueueIndex(size)].template allocPage<OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(this);

}

// ##############################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
__forceinline__ __device__ void OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::freePage(MemoryIndex index)
{
	if(statistics_enabled)
		atomicAdd(&stats.pageFreeCount, 1);

	// Deallocate page in chunk
	d_storage_reuse_queue[QueueType::ChunkType::template getQueueIndexFromPage<QI>(d_data, start_index, index.getChunkIndex())].freePage(this, index);
}

// ##############################################################################################################################################
//
//
// OUROBOROS
//
//
// ##############################################################################################################################################

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
__forceinline__ __device__ void* Ouroboros<OUROBOROS, OUROBOROSES...>::malloc(size_t size)
{
	if(size <= ConcreteOuroboros::LargestPageSize_)
	{
		return memory_manager.allocPage(size);
	}
	return next_memory_manager.malloc(size);
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
__forceinline__ __device__ void Ouroboros<OUROBOROS, OUROBOROSES...>::free(void* ptr)
{
	if(!validOuroborosPointer(ptr))
	{
		if(printDebug)
			printf("Freeing CUDA Memory!\n");
		free(adjacency);
		return;
	}

	auto chunk_index = CommonChunk::getIndexFromPointer(memory.d_data, memory.start_index, ptr);
	auto chunk = reinterpret_cast<CommonChunk*>(ConcreteOuroboros::ChunkBase::getMemoryAccess(memory.d_data, memory.start_index, chunk_index));
	auto page_size = chunk->page_size;
	auto page_index = CommonChunk::getPageIndexFromPointer(memory.d_data, memory.start_index, ptr, page_size)
	index.getIndex(chunk_index, page_index);
	return freePageRecursive(page_size, index);
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
__forceinline__ __device__ void Ouroboros<OUROBOROS, OUROBOROSES...>::freePageRecursive(unsigned int page_size, MemoryIndex index)
{
	if(page_size <= ConcreteOuroboros::LargestPageSize_)
	{
		return memory_manager.freePage(index);
	}
	return next_memory_manager.freePageRecursive(page_size, index);
}

// ##############################################################################################################################################
//
//
// HOST
//
//
// ##############################################################################################################################################

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
void updateMemoryManagerHost(MemoryManagerType& memory_manager)
{
	HANDLE_ERROR(cudaMemcpy(&memory_manager,
		memory_manager.memory.d_memory,
		sizeof(memory_manager),
		cudaMemcpyDeviceToHost));
}

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
void updateMemoryManagerDevice(MemoryManagerType& memory_manager)
{
	HANDLE_ERROR(cudaMemcpy(memory_manager.memory.d_memory,
		&memory_manager,
		sizeof(memory_manager),
		cudaMemcpyHostToDevice));
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
Ouroboros<OUROBOROS, OUROBOROSES...>::~Ouroboros()
{
	if(printDebug)
		printf("Ouroboros Destructor called\n");
	
	updateMemoryManagerHost(*this);

	if (memory.d_memory != nullptr)
	{
		cudaFree(memory.d_memory);
		memory.d_memory = nullptr;
		memory.d_data = nullptr;
		memory.d_data_end = nullptr;
	}
}