#pragma once

#include "BulkSemaphore.cuh"

// ##############################################################################################################################################
//
__forceinline__ __device__ bool BulkSemaphore::tryReduce(int N)
{
	// Reduce by N-1
	uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(N)) & highest_value_mask;
	if((atomic_ret_val - N) < null_value)
	{
		atomicAdd(&value, N);
		return false;
	}
	return true;
}

// ##############################################################################################################################################
//
#if (__CUDA_ARCH__ < 700)
template <typename T>
__forceinline__ __device__ void BulkSemaphore::wait(int N, uint32_t number_pages_on_chunk, T allocationFunction)
{
	enum class Mode
	{
		AllocateChunk, AllocatePage, Reserve, Invalid
	};
	auto mode{ Mode::Invalid };
	while(true)
	{
		if(getCount() - N >= 0)
		{
			// Try to decrement global count first
			uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(N)) & highest_value_mask;
			if((atomic_ret_val - N) >= null_value)
			{
				return;
			}
			// Increment count again
			atomicAdd(&value, N);
		}	

		BulkSemaphore old_semaphore_value;
		int expected, reserved, count;
		
		// Read from global
		BulkSemaphore new_semaphore_value{ Ouro::ldg_cg(&value) };
		do
		{
			old_semaphore_value = new_semaphore_value;
			new_semaphore_value.getValues(count, expected, reserved);

			if ((count + expected - reserved) < N)
			{
				// We are the one to allocate
				expected += number_pages_on_chunk;
				mode = { Mode::AllocateChunk };
			}
			else if (count >= N)
			{
				// We have enough resources available to satisfy this request
				count -= N;
				mode = { Mode::AllocatePage };
			}
			else
			{
				// Not enough here, let's reserve some stuff and wait
				reserved += N;
				mode = { Mode::Reserve };
			}

			new_semaphore_value.createValueInternal(count, expected, reserved);
			// Try to set new value
		} while ((new_semaphore_value.value = atomicCAS(&value, old_semaphore_value.value, new_semaphore_value.value))
			!= old_semaphore_value.value);

		// ##############################################
		// Return if chunk allocation or page allocation
		if (mode == Mode::AllocatePage)
			return;

		int predicate = (mode == Mode::AllocateChunk) ? 1 : 0;
		if (__ballot_sync(__activemask(), predicate))
		{
			if(predicate)
			{
				allocationFunction();
			}
		}
		__syncwarp();
		if(mode == Mode::Reserve)
		{
			// ##############################################
			// Wait on our resource
			unsigned int counter {0U};
			do
			{
				// Yield for some time
				Ouro::sleep(++counter);

				// Read from global
				read(new_semaphore_value);
				new_semaphore_value.getValues(count, expected, reserved);
			} while ((count < N) && (reserved < (count + expected)));

			// ##############################################
			// Reduce reserved count
			atomicAdd(&value, create64BitSubAdder_reserved(N));
		}
	}
}
#else
// ##############################################################################################################################################
//
template <typename T>
__forceinline__ __device__ void BulkSemaphore::wait(int N, uint32_t number_pages_on_chunk, T allocationFunction)
{
	enum class Mode
	{
		AllocateChunk, AllocatePage, Reserve, Invalid
	};
	auto mode{ Mode::Invalid };
    int mask = __match_any_sync(__activemask(), number_pages_on_chunk);
	int leader = __ffs(mask) - 1; // Select leader
		
	int leader_mask = __match_any_sync(__activemask(), Ouro::lane_id() == leader);
	if(Ouro::lane_id() == leader)
	{
		int num = __popc(mask) * N; // How much should our allocator allocate?
		while(true)
		{
			if(getCount() - static_cast<int>(num) >= 0)
			{
				uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(num)) & highest_value_mask; // Try to decrement global count first
				if((atomic_ret_val - num) >= null_value)
				{
					break;
				}

				__threadfence_block();

				// Increment count again
				atomicAdd(&value, num);
			}
			
			BulkSemaphore old_semaphore_value;
			int expected, reserved, count;
			// Read from global
			BulkSemaphore new_semaphore_value{ Ouro::ldg_cg(&value) };
			do
			{
				old_semaphore_value = new_semaphore_value;
				new_semaphore_value.getValues(count, expected, reserved);

				if ((count + expected - reserved) < num)
				{
					// We are the one to allocate
					expected += number_pages_on_chunk;
					mode = { Mode::AllocateChunk };
				}
				else if (count >= num)
				{
					// TODO: Can we maybe try this with an atomic?
					// We have enough resources available to satisfy this request
					count -= num;
					mode = { Mode::AllocatePage };
				}
				else
				{
					// Not enough here, let's reserve some stuff and wait
					reserved += num;
					mode = { Mode::Reserve };
				}

				new_semaphore_value.createValueInternal(count, expected, reserved);
				// Try to set new value
			}
			while ((new_semaphore_value.value = atomicCAS(&value, old_semaphore_value.value, new_semaphore_value.value))
			!= old_semaphore_value.value);

			if (mode == Mode::AllocatePage)
				break;

			if(mode == Mode::AllocateChunk)
			{
				allocationFunction();
			}
			// TODO: Not needed??
			__syncwarp(leader_mask);
			// __threadfence_block();

			if(mode == Mode::Reserve)
			{
				// ##############################################
				// Wait on our resource
				unsigned int counter {0U};
				do
				{
					// Yield for some time
					Ouro::sleep(++counter);

					// Read from global
					read(new_semaphore_value);
					new_semaphore_value.getValues(count, expected, reserved);
				} 
				while ((count < num) && (reserved < (count + expected)));

				// ##############################################
				// Reduce reserved count
				atomicAdd(&value, create64BitSubAdder_reserved(num));
			}
		}
	}
	// Gather all threads around -> wait for the leader to finish
	__syncwarp(mask);
}
#endif

// ##############################################################################################################################################
//
__forceinline__ __device__ int BulkSemaphore::signalExpected(unsigned long long N)
{
	return static_cast<int>(atomicAdd(&value, N + create64BitSubAdder_expected(N)) & highest_value_mask) - null_value;
}

// ##############################################################################################################################################
//
__forceinline__ __device__ int BulkSemaphore::signal(unsigned long long N)
{
	return static_cast<int>(atomicAdd(&value, N) & highest_value_mask) - null_value;
}