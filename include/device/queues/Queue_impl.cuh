#pragma once
#include "Queue.h"
#include "Utility.cuh"

__forceinline__ __host__ void IndexQueue::resetQueue()
{
	count_ = 0;
	front_ = 0;
	back_ = 0;
}

__forceinline__ __device__ void IndexQueue::init()
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
	{
		queue_[i] = DeletionMarker<index_t>::val;
	}
}

__forceinline__ __device__ bool IndexQueue::enqueue(index_t i)
{
	int fill = atomicAdd(&count_, 1);
	if (fill < static_cast<int>(size_))
	{
		//we have to wait in case there is still something in the spot
		// note: as the filllevel could be increased by this thread, we are certain that the spot will become available
		unsigned int pos = atomicAdd(&back_, 1) % size_;
		while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, i) != DeletionMarker<index_t>::val)
			Ouro::sleep();
			
		return true;
	}
	else
	{
		//__trap(); //no space to enqueue -> fail
		return false;
	}
}

template <int CHUNK_SIZE>
__forceinline__ __device__ bool IndexQueue::enqueueClean(index_t i, index_t* chunk_data_ptr)
{
	for(auto i = 0U; i < (CHUNK_SIZE / (sizeof(index_t))); ++i)
	{
		atomicExch(&chunk_data_ptr[i], DeletionMarker<index_t>::val);
	}

	__threadfence_block();

	// Enqueue now
	return enqueue(i);
}

__forceinline__ __device__ int IndexQueue::dequeue(index_t& element)
{
	int readable = atomicSub(&count_, 1);
	if (readable <= 0)
	{
		//dequeue not working is a common case
		atomicAdd(&count_, 1);
		return FALSE;
	}
	unsigned int pos = atomicAdd(&front_, 1) % size_;
	while ((element = atomicExch(queue_ + pos, DeletionMarker<index_t>::val)) == DeletionMarker<index_t>::val)
		Ouro::sleep();
	return TRUE;
}