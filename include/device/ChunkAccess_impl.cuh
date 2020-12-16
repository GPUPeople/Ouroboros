#pragma once

#include "device/ChunkAccess.cuh"

// ##############################################################################################################################################
//
template <size_t SIZE, size_t SMALLEST_PAGE>
__forceinline__ __device__ ChunkAccess<SIZE, SMALLEST_PAGE>::FreeMode ChunkAccess<SIZE, SMALLEST_PAGE>::freePage(index_t page_index)
{
	const int mask_index = page_index / (Ouro::sizeofInBits<MaskDataType>());
	const int local_page_index = page_index % (Ouro::sizeofInBits<MaskDataType>());
	const auto bit_pattern = 1U << local_page_index;
	// Set bit to 1
	atomicOr(&availability_mask[mask_index], bit_pattern);
	
	// Please do NOT reorder here
	__threadfence_block();

	auto current_count = atomicAdd(&count, 1) + 1;
	if (current_count == 1)
		return FreeMode::FIRST_FREE;
	else if(current_count == size)
		return FreeMode::DEQUEUE;
	return FreeMode::SUCCESS;
}

// ##############################################################################################################################################
//
template <size_t SIZE, size_t SMALLEST_PAGE>
__forceinline__ __device__ bool ChunkAccess<SIZE, SMALLEST_PAGE>::tryFlashChunk()
{
	// Try to reduce count to 0, if previous value is != size, someone tries do allocate from this chunk right now!
	return atomicCAS(&count, size, 0) == size;
}

// ##############################################################################################################################################
//
template <size_t SIZE, size_t SMALLEST_PAGE>
__forceinline__ __device__ ChunkAccess<SIZE, SMALLEST_PAGE>::Mode ChunkAccess<SIZE, SMALLEST_PAGE>::allocPage(index_t& page_index)
{
	int current_count{ 0 };
	auto mode = Mode::SUCCESSFULL;
	while ((current_count = atomicSub(&count, 1)) <= 0)
	{
		if((current_count = atomicAdd(&count, 1)) < 0)
			return Mode::CONTINUE;

		// If we observed 0 -> 1, we potentially want to re-enqueue this chunk
		if(current_count == 0)
			mode = Mode::RE_ENQUEUE_CHUNK;
	}
	if(current_count == 1)
	{
		// We take the last page (so just take the page and don't re-enqueue) 
		// OR 
		// We just want to dequeue this chunk
		// TODO: Not sure if this logic is completely right, if not both modi need the DEQUEUE_CHUNK
		// mode = (mode == Mode::RE_ENQUEUE_CHUNK) ? Mode::SUCCESSFULL : Mode::DEQUEUE_CHUNK;
		mode = Mode::DEQUEUE_CHUNK;
	}
	// else
	// {
	// 	// If we had a count larger than 1, we can either simply do as normal or stay in the re-enqueue mode
	// 	if(mode != Mode::RE_ENQUEUE_CHUNK)
	// 		mode = Mode::SUCCESSFULL;
	// }

	int least_significant_bit{ 0 };

	// Offset in the range of 0-63
	const int offset = (threadIdx.x + blockIdx.x) % Ouro::sizeofInBits<MaskDataType>();

	// TODO: Why is this not faster instead of always using the full mask?
	// int mask = Ouro::divup(size, sizeof(MaskDataType) * BYTE_SIZE);
	int bitmask_index = threadIdx.x;

	// There is a reason why this is a while true loop and not just a loop over all MAXIMUM_MITMASK_SIZE entries
	// Imagine we have 2 threads, currently one page in mask 3
	// One thread decrements the count and starts at the first mask to look for the bit -> does not find any
	// One thread then frees a page on the first mask
	// A second thread decrements the count and starts looking at mask 3 -> finds the bit immediately
	// The first thread would now look at mask 2 - 3 - 4 ... and not find the bit on mask 1, as it already looked there
	// Hence, we need a while(true) loop, since we are guaranteed to find a bit, but not guaranteed that someone steals our bit
	// unsigned int iters{0U};
	while(true)
	{
		// if(++iters >= 10000000)
		// {
		// 	printf("%d - %d | On Chunk: %u ---  What the hello with current count received: %u\n", blockIdx.x, threadIdx.x, chunk_id, current_count);
		// 	__trap();
		// }

		// We want each thread starting at a different position, for this we do a circular shift
		// This way we can still use the build in __ffsll but will still start our search at different 
		// positions
		// Load mask -> shift by offset to the right and then append whatever was shifted out at the top
		auto current_mask = Ouro::ldg_cg(&availability_mask[(++bitmask_index) % MaximumBitMaskSize_]);
		auto without_lower_part = current_mask >> offset;
		auto final_mask = without_lower_part | (current_mask << (Ouro::sizeofInBits<MaskDataType>() - offset));

		while(least_significant_bit = __ffsll(final_mask))
		{
			--least_significant_bit; // Get actual bit position (as bit 0 return 1)
			least_significant_bit = ((least_significant_bit + offset) % Ouro::sizeofInBits<MaskDataType>()); // Correct for shift
			page_index = Ouro::sizeofInBits<MaskDataType>() * (bitmask_index % MaximumBitMaskSize_) // which mask
				+ least_significant_bit; // which page on mask

			// Please do NOT reorder here
			__threadfence_block();

			auto bit_pattern = createBitPattern(least_significant_bit);
			current_mask = atomicAnd(&availability_mask[bitmask_index % MaximumBitMaskSize_], bit_pattern);

			// Please do NOT reorder here
			__threadfence_block();

			if(checkBitSet(current_mask, least_significant_bit))
			{
				// Hehe, we were the one who set this bit :-)
				return mode;
			}
			without_lower_part = current_mask >> offset;
			final_mask = without_lower_part | (current_mask << (Ouro::sizeofInBits<MaskDataType>() - offset));
		}
	}

	// ##############################################################################################################
	// Error Checking
	if(!FINAL_RELEASE)
	{
		printf("We should have gotten a page, but there was nothing for threadId %d and blockId %d - current count : %d - bitmask_index: %d\n", threadIdx.x, blockIdx.x, current_count, bitmask_index);
		__trap();
	}
	return Mode::ERROR;
}
