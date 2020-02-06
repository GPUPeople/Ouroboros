#pragma once

#include "device/MemoryIndex.cuh"
#include "device/queues/Queues.cuh"
#include "Statistics.h"
#include "Helper.cuh"

struct Memory
{
	// Data
	memory_t* d_memory{ nullptr };
	memory_t* d_data{ nullptr };
	memory_t* d_data_end{ nullptr };
	uint64_t start_index;
	index_t next_free_chunk{ 0 };

	size_t maxChunks{ 0 };
	size_t allocationSize{ 0 };
	size_t adjacencysize{ 0 };
	size_t additionalSizeBeginning{0};
	size_t additionalSizeEnd{0};
};