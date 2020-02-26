#pragma once

#include "Utility.h"

struct ChunkLocator
{
	// Static Members
	static constexpr int division_factor{5}; // Divide by 32

	// Member
	int* d_chunk_flags{nullptr};

	__device__ __forceinline__ void init(unsigned int num_chunks)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_chunks; i += blockDim.x * gridDim.x)
		{
			d_chunk_flags[i] = 0;
		}
	}

	__device__ __forceinline__ unsigned int getChunkIndex(unsigned int chunk_index)
	{
		auto index = chunk_index >> division_factor; // Get index position
		auto mask = (1 << Ouro::modPower2<32>(chunk_index)) - 1;
		while(true)
		{
			auto flags = d_chunk_flags[index];
			auto local_index = 32 - __clz(flags & mask);
			if(local_index)
				return (index << division_factor) + local_index;

			// Go back
			--index;
			mask = 0xFFFFFFFF;
		}
		
	}
};