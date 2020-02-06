#pragma once

#include "Definitions.h"

class IndexQueue
{
public:
	__forceinline__ __device__ void init();

	__forceinline__ __device__ bool enqueue(index_t i);

	template <int CHUNK_SIZE>
	__forceinline__ __device__ bool enqueueClean(index_t i, index_t* chunk_data_ptr);

	__forceinline__ __device__ int dequeue(index_t& element);

	void resetQueue();

	index_t* queue_;
	int count_{ 0 };
	unsigned int front_{ 0 };
	unsigned int back_{ 0 };
	int size_{ 0 };
};