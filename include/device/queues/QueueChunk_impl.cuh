#pragma once

#include "device/queues/QueueChunk.cuh"

// ##############################################################################################################################################
// Is instantiated 3 times for enqueue, enqueueChunk and dequeue
template <typename ChunkBase>
template <typename FUNCTION>
__forceinline__ __device__ void QueueChunk<ChunkBase>::guaranteeWarpSyncPerChunk(index_t position, const char* message, FUNCTION f)
{
	// This functions tries to guarantee that threads that concurrently do a certain action will be synchronized in their traversal
	// Otherwise the issue is that, although the new model seems to state otherwise, certain threads can be stalled by others
	// Which might mean that traversal is hindered and the whole model hangs
	QueueChunk<ChunkBase>* chunk_ptr{this};
	int work_NOT_done{TRUE};

	// TODO: This seems to work, but it is not a guarantee that this actually works
	auto active_mask = __activemask();
	while(true)
	{
		int predicate = (chunk_ptr->checkVirtualStart(position)) ? work_NOT_done : FALSE;
		if (__any_sync(active_mask, predicate))
		{
			if(predicate)
			{
				// Execute function (enqueue | enqueueChunk | dequeue)
				f(chunk_ptr);

				__threadfence_block();

				// Now our work is done
				work_NOT_done = FALSE;
			}
		}

		// Guarantee that no currently active threads continues to the traversal before the allocation from before is done
		__syncwarp(active_mask);

		// Is there still someone with work left or not?
		if (__any_sync(active_mask, work_NOT_done))
		{
			// This is critical for threads within a warp that operate on different queues
			// Only threads which still have to do their work, hence expect a new chunk to arrive, should try to traverse
			// The others, which are done, might already be at the last chunk for their queue -> they might hang here!
			if(work_NOT_done)
			{
				// Continue traversal
				unsigned long long next {DeletionMarker<unsigned long long>::val};
				unsigned int counter{0};
				// Next might not be set yet, in this case we have to wait
				while((next = Ouro::ldg_cg(&chunk_ptr->next_)) == DeletionMarker<unsigned long long>::val)
				{
					if(counter++ > (1000*1000*10))
					{
						if(!FINAL_RELEASE)
							printf("%d : %d died in gWS from %s index: %u - ptr: %p | mask: %x\n", threadIdx.x, blockIdx.x, message, chunk_ptr->chunk_index_, chunk_ptr, active_mask);
						__trap();
					}
					Ouro::sleep(counter);
				}

				// Do NOT reorder here
				__threadfence_block();

				// Continue to next chunk and check there again
				chunk_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(next);
			}
		}
		else
			break;
	}
}

// ##############################################################################################################################################
// 
template <typename ChunkBase>
__forceinline__ __device__ void QueueChunk<ChunkBase>::enqueueInitial(const unsigned int position, const QueueDataType element)
{
	queue_[position] = element;
	// Increment both counters
	count_ += countAddValueEnqueue<1>();
	//printf("%d - %d | Position: %u - Element: %u - Enqueue initial: %u - %u | %p\n", threadIdx.x, blockIdx.x, position, element, extractCounterA(count_), extractCounterB(count_), queue_);
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ unsigned int QueueChunk<ChunkBase>::enqueue(const unsigned int position, const QueueDataType element)
{
	unsigned int counter{0};
	unsigned int test_val{0U};
	while ((test_val = atomicCAS(queue_ + position, DeletionMarker<QueueDataType>::val, element)) != DeletionMarker<QueueDataType>::val)
	{
		// TODO: Change this back!
		// Ouro::sleep(counter);
		Ouro::sleep();
		if(++counter > (1000*1000*10))
		{
			if (!FINAL_RELEASE)
				printf("%d - %d | Horrible death in enqueue: Position: %u - Chunk Index: %u -> Value: %u | %p\n", threadIdx.x, blockIdx.x, position, chunk_index_, test_val, queue_);
			__trap();
		}
	}

	// Increment both counters
	return atomicAdd(&count_, countAddValueEnqueue<1>());
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ unsigned int QueueChunk<ChunkBase>::enqueueLinked(const unsigned int position, const QueueDataType element)
{
	atomicExch(queue_ + position, element);

	// Increment both counters
	return atomicAdd(&count_, countAddValueEnqueue<1>());
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ unsigned int QueueChunk<ChunkBase>::enqueueLinkedv4(const unsigned int position, const index_t chunk_index, const index_t start_index)
{
	uint4 indices{
		MemoryIndex::createIndex(chunk_index, start_index),
		MemoryIndex::createIndex(chunk_index, start_index + 1),
		MemoryIndex::createIndex(chunk_index, start_index + 2),
		MemoryIndex::createIndex(chunk_index, start_index + 3)
	};
	Ouro::store(reinterpret_cast<uint4*>(queue_ + position), indices);
	return atomicAdd(&count_, countAddValueEnqueue<4>());
}

// ##############################################################################################################################################
// Enqueue a single page/chunk into the queue
template <typename ChunkBase>
template <typename MemoryManagerType>
__forceinline__ __device__ void QueueChunk<ChunkBase>::enqueue(MemoryManagerType* memory_manager, const unsigned int position, const QueueDataType element, QueueChunk<ChunkBase>** queue_next_ptr, QueueChunk<ChunkBase>** queue_front_ptr, QueueChunk<ChunkBase>** queue_old_ptr, unsigned int* old_count)
{
	guaranteeWarpSyncPerChunk(position, "Enqueue", [&](QueueChunk<ChunkBase>* chunk_ptr)
	{
		// We found the right chunk
		const auto local_position = (Ouro::modPower2<num_spots_>(position));
		if(local_position == 0)
		{
			// We pre-emptively allocate the next chunk already
			unsigned int chunk_index{ 0 };
			memory_manager->allocateChunk<true>(chunk_index);

			if(!FINAL_RELEASE && printDebug)
				printf("E - %d : %d Allocate a new chunk for queue with index: %u and virtual pos: %u\n", threadIdx.x, blockIdx.x, chunk_index, position);

			auto potential_next = initializeChunk(memory_manager->d_data, chunk_index, position + num_spots_);

			// Let everyone hear about the joyful news, a new chunk has been born
			__threadfence();

			atomicExch(&chunk_ptr->next_, reinterpret_cast<unsigned long long>(potential_next));
			if(!FINAL_RELEASE && printDebug)
				printf("E - %d : %d Allocate a new chunk for the queue with index: %u and address: %p\n", threadIdx.x, blockIdx.x, chunk_index, potential_next);
		}

		// Do NOT reorder here
		__threadfence_block();

		// Finally enqueue on this chunk - get back counter and just take lower 32bit -> counterA | atomic returns value before add so we need + 1
		auto current_count = chunk_ptr->enqueueLinked(local_position, element);
		if(extractCounterA(current_count) + 1 == num_spots_)
		{
			// This chunk is now full, we can move the back pointer ahead to the next chunk
			chunk_ptr->setBackPointer(queue_next_ptr);
		}
		if(checkChunkEmptyEnqueue<1>(current_count))
		{
			// We can remove this chunk
			auto how_many_removed = chunk_ptr->setFrontPointer(queue_front_ptr);
			
			// Do NOT reorder here (but since the next call depends on the previous, this would be quite stupid)
			__threadfence_block();

			chunk_ptr->setOldPointer(memory_manager, queue_old_ptr, old_count, how_many_removed);
		}
	});
}

// ##############################################################################################################################################
// Enqueue a full chunk into the queue (in this case many pages from a chunk)
template <typename ChunkBase>
template <typename MemoryManagerType>
__forceinline__ __device__ void QueueChunk<ChunkBase>::enqueueChunk(MemoryManagerType* memory_manager, unsigned int position, index_t chunk_index, index_t pages_per_chunk, QueueChunk<ChunkBase>** queue_next_ptr, QueueChunk<ChunkBase>** queue_front_ptr, QueueChunk<ChunkBase>** queue_old_ptr, unsigned int* old_count, int start_index)
{
	guaranteeWarpSyncPerChunk(position, "EnqueueChunk", [&](QueueChunk<ChunkBase>* chunk_ptr)
	{
		// First check if we have to allocate an additional queue chunk at this point?
		QueueChunk<ChunkBase>* potential_next{nullptr};
		unsigned int queue_chunk_index{ 0 };
		auto local_position = (Ouro::modPower2<num_spots_>(position));
		if(local_position == 0 || ((local_position + pages_per_chunk) > num_spots_))
		{
			// In this case we are either directly the first on a chunk or we will wrap onto the new chunk and then be first
			// In either case, pre-allocate a new chunk here
			memory_manager->allocateChunk<true>(queue_chunk_index);

			if(!FINAL_RELEASE && printDebug)
				printf("EC - %d : %d Allocate a new chunk for queue with index: %u and virtual pos: %u\n", threadIdx.x, blockIdx.x, queue_chunk_index, (local_position == 0) ? (position + num_spots_) : (position + num_spots_ + (num_spots_ - local_position)) );

			index_t next_virtual_start{ (local_position == 0) ? (position + num_spots_) : (position + num_spots_ + (num_spots_ - local_position)) };
			potential_next = initializeChunk(memory_manager->d_data, queue_chunk_index, next_virtual_start);

			// Let everyone hear about the joyful news, a new chunk has been born
			__threadfence();
		}

		//Mode mode {Mode::SINGLE};
		index_t current_index_offset = 0;
		Mode mode {((Ouro::modPower2<vector_width>(local_position) == 0) && (pages_per_chunk >= vector_width)) ? Mode::V4 : Mode::SINGLE};

		// Now we want to insert pages_per_chunk into our queue
		while(current_index_offset < pages_per_chunk)
		{
			if(local_position == 0)
			{
				// In this case we can set our current next pointer to the previously allocated chunk
				atomicExch(&chunk_ptr->next_, reinterpret_cast<unsigned long long>(potential_next));
				if(!FINAL_RELEASE && printDebug)
					printf("EC - %d : %d Allocate a new chunk for the queue with index: %u and address: %p\n", threadIdx.x, blockIdx.x, queue_chunk_index, potential_next);
			}

			// Enqueue the element at this position (we don't have to wait as enqueue does not have a problem with dequeue as with normal queue -> no wrap around)
			unsigned int current_count{0};
			if(mode == Mode::SINGLE)
			{
				current_count = chunk_ptr->enqueueLinked(local_position, MemoryIndex::createIndex(chunk_index, current_index_offset + start_index));
				if(extractCounterA(current_count) + 1 == num_spots_)
				{
					// This chunk is now full, we can move the back pointer ahead to the next chunk
					chunk_ptr->setBackPointer(queue_next_ptr);
				}
				if(checkChunkEmptyEnqueue<1>(current_count))
				{
					// We can remove this chunk
					auto how_many_removed = chunk_ptr->setFrontPointer(queue_front_ptr);
					
					// Do NOT reorder here (but since the next call depends on the previous, this would be quite stupid)
					__threadfence_block();

					chunk_ptr->setOldPointer(memory_manager, queue_old_ptr, old_count, how_many_removed);
				}
			}
			else
			{
				current_count = chunk_ptr->enqueueLinkedv4(local_position, chunk_index, current_index_offset + start_index);
				if(extractCounterA(current_count) + vector_width == num_spots_)
				{
					// This chunk is now full, we can move the back pointer ahead to the next chunk
					chunk_ptr->setBackPointer(queue_next_ptr);
				}
				if(checkChunkEmptyEnqueue<vector_width>(current_count))
				{
					// We can remove this chunk
					auto how_many_removed = chunk_ptr->setFrontPointer(queue_front_ptr);
					
					// Do NOT reorder here (but since the next call depends on the previous, this would be quite stupid)
					__threadfence_block();

					chunk_ptr->setOldPointer(memory_manager, queue_old_ptr, old_count, how_many_removed);
				}
			}

			// Do NOT reorder here
			__threadfence_block();

			// If we are at the end, we want to traverse and set the back pointer
			if(goToNextChunk(local_position, mode))
			{
				// Continue to next chunk
				unsigned long long next {DeletionMarker<unsigned long long>::val};
				unsigned int counter{0};
				// Next might not be set yet, in this case we have to wait
				while((next = Ouro::ldg_cg(&chunk_ptr->next_)) == DeletionMarker<unsigned long long>::val)
				{
					if(counter++ > (1000*1000*10))
					{
						if(!FINAL_RELEASE)
							printf("%d : %d died in traversal in enqueueChunk, chunk_index: %u - ptr: %p\n", threadIdx.x, blockIdx.x, chunk_ptr->chunk_index_, chunk_ptr);
						__trap();
					}
					Ouro::sleep(counter);
				}

				// Do NOT reorder here
				__threadfence_block();

				// Continue to next chunk and check there again
				chunk_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(next);
				local_position = 0;
			}
			else
			{
				local_position += enqueueChunkAdditionFactor(mode);
			}

			current_index_offset += enqueueChunkAdditionFactor(mode);
			
			// Compute mode for next round
			if(mode == Mode::SINGLE)
			{
				if((Ouro::modPower2<vector_width>(local_position) == 0) && ((pages_per_chunk - current_index_offset) >= vector_width))
					mode = Mode::V4;
			}
			else
			{
				if((pages_per_chunk - current_index_offset) < vector_width)
					mode = Mode::SINGLE;
			}
		}
	});
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
template <typename MemoryManagerType>
__forceinline__ __device__ bool QueueChunk<ChunkBase>::dequeue(const unsigned int position, QueueDataType& element, MemoryManagerType* memory_manager, QueueChunk<ChunkBase>** queue_front_ptr)
{
	unsigned int counter{0};
	// Element might currently not yet be present (enqueue advertised it already, but has not put element in) -> spin on value!
	while ((element = atomicExch(queue_ + position, DeletionMarker<QueueDataType>::val)) == DeletionMarker<QueueDataType>::val)
	{
		Ouro::sleep(counter);
		if(++counter > (1000*1000*10))
		{
			if (!FINAL_RELEASE)
				printf("%d : %d | Horrible death in dequeue: Position: %u - Chunk Index: %u - virtual_pos: %u - frontpointerstart: %u\n", threadIdx.x, blockIdx.x, position, chunk_index_, virtual_start_ + position, (queue_front_ptr)?(*queue_front_ptr)->virtual_start_ : 0);

			__trap();
		}
	}

	// We have taken our element out - Return true if this chunk is empty and can be removed from the chunk list
	// We subtract from counterB, so at this point, if counterB is 0 and counterA is full, then this chunk is empty
	// Since atomicAdd returns the old value, check if the result is equal to counterA = num_spots and counterB = 1
	return checkChunkEmptyDequeue(atomicSub(&count_, 1 << shift_value));
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ bool QueueChunk<ChunkBase>::deleteElement(const unsigned int position)
{
	// Since we don't care about the value, we can simply delete it (no matter if enqueue might delete it as well even later)
	atomicExch(&queue_[position], DeletionMarker<QueueDataType>::val);

	// We have taken our element out - Return true if this chunk is empty and can be removed from the chunk list
	// We subtract from counterB, so at this point, if counterB is 0 and counterA is full, then this chunk is empty
	// Since atomicAdd returns the old value, check if the result is equal to counterA = num_spots and counterB = 1
	return checkChunkEmptyDequeue(atomicSub(&count_, 1U << shift_value));
}

// ##############################################################################################################################################
// Dequeue a page or chunk index from the queue
template <typename ChunkBase>
template <typename QueueChunk<ChunkBase>::DEQUEUE_MODE Mode, typename MemoryManagerType>
__forceinline__ __device__ void QueueChunk<ChunkBase>::dequeue(MemoryManagerType* memory_manager, const unsigned int position, QueueDataType& element, QueueChunk<ChunkBase>** queue_front_ptr, QueueChunk<ChunkBase>** queue_old_ptr, unsigned int* old_count)
{
	guaranteeWarpSyncPerChunk(position, "Dequeue", [&](QueueChunk<ChunkBase>* chunk_ptr)
	{
		const auto local_position = (Ouro::modPower2<num_spots_>(position));
		if(Mode == DEQUEUE_MODE::DEQUEUE ? chunk_ptr->dequeue(local_position, element, memory_manager, queue_front_ptr) : chunk_ptr->deleteElement(local_position))
		{
			// We can remove this chunk
			auto how_many_removed = chunk_ptr->setFrontPointer(queue_front_ptr);
			
			// Do NOT reorder here (but since the next call depends on the previous, this would be quite stupid)
			__threadfence_block();

			chunk_ptr->setOldPointer(memory_manager, queue_old_ptr, old_count, how_many_removed);
		}
	});
	return;
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ QueueChunk<ChunkBase>* QueueChunk<ChunkBase>::locateQueueChunkForPosition(const unsigned int v_position, const char* message)
{
	// Start at current chunk
	QueueChunk<ChunkBase>* chunk_ptr{this};
	// Check if this is already the correct chunk
	while(!chunk_ptr->checkVirtualStart(v_position))
	{
		// Continue to next chunk
		unsigned long long next {DeletionMarker<unsigned long long>::val};
		unsigned int counter{0};
		// Next might not be set yet, in this case we have to wait
		while((next = Ouro::ldg_cg(&chunk_ptr->next_)) == DeletionMarker<unsigned long long>::val)
		{
			if(counter++ > (1000*1000*10))
			{
				if(!FINAL_RELEASE)
					printf("%d : %d died in LocateQueueChunk in virtual start, coming from %s, chunk_index: %u - ptr: %p\n", threadIdx.x, blockIdx.x, message, chunk_ptr->chunk_index_, chunk_ptr);
				__trap();
			}
			Ouro::sleep(counter);
		}

		// Do NOT reorder here
		__threadfence_block();

		// Continue to next chunk and check there again
		chunk_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(next);
	}

	return chunk_ptr;
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ void QueueChunk<ChunkBase>::accessLinked(const unsigned position, QueueDataType& element)
{
	// Traverse to correct chunk and then access queue_ at correct position
	QueueChunk<ChunkBase>* current_chunk{locateQueueChunkForPosition(position, "ACCESSLINKED")};
	element = Ouro::ldg_cg(&(current_chunk->queue_[Ouro::modPower2<num_spots_>(position)]));
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ QueueChunk<ChunkBase>* QueueChunk<ChunkBase>::accessLinked(const unsigned position)
{
	// Traverse to correct chunk and then access queue_ at correct position
	return locateQueueChunkForPosition(position, "ACCESSLINKED");
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ void QueueChunk<ChunkBase>::setBackPointer(QueueChunk<ChunkBase>** queue_next_ptr)
{
	// INFO: setNextPointer is only called, if all spots on this chunk called their enqueue, at which point next_ must have been set already
	QueueChunk<ChunkBase>* chunk_ptr{this};
	// Try to set back pointer with current chunks next pointer, continue in loop if successful!
	while(atomicCAS((reinterpret_cast<unsigned long long*>(queue_next_ptr)), reinterpret_cast<unsigned long long>(chunk_ptr), Ouro::ldg_cg(&chunk_ptr->next_)) 
		== reinterpret_cast<unsigned long long>(chunk_ptr))
	{
		if(!FINAL_RELEASE && printDebug)
			printf("%d : %d Moved backpointer from virtual start: %u to %u\n", threadIdx.x, blockIdx.x, chunk_ptr->virtual_start_, reinterpret_cast<QueueChunk<ChunkBase>*>(chunk_ptr->next_)->virtual_start_);
		// Read the count of the next chunk in line, check if counterA is already full as well
		auto current_count_A = extractCounterA(Ouro::ldg_cg(&reinterpret_cast<QueueChunk<ChunkBase>*>(chunk_ptr->next_)->count_));
		// If counterA is full, we want to try to continue advancing the back pointer, since multiple chunks might be full at the same time
		// In this case we still want to move our pointer over all, if not we can break here
		if(current_count_A == num_spots_)
		{
			// Set pointer to next pointer
			chunk_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(Ouro::ldg_cg(&chunk_ptr->next_));
		}
		else
			break;
	}
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
__forceinline__ __device__ unsigned int QueueChunk<ChunkBase>::setFrontPointer(QueueChunk<ChunkBase>** queue_front_ptr)
{
	// INFO: We can only remove a queue_chunk, if beforehand all enqueues have been done, so "next_" has to exist as well!
	QueueChunk<ChunkBase>* chunk_ptr{this};
	unsigned int ret_val{0};
	// Try to set front pointer with current chunks next pointer, continue in loop if successfull
	while(atomicCAS((reinterpret_cast<unsigned long long*>(queue_front_ptr)), reinterpret_cast<unsigned long long>(chunk_ptr), Ouro::ldg_cg(&chunk_ptr->next_)) 
		== reinterpret_cast<unsigned long long>(chunk_ptr))
	{
		++ret_val;

		// We check if the count of the next chunk is equal to num_spots (counterA = num_spots & counterB = 0) -> this chunk is empty as well
		if(Ouro::ldg_cg(&reinterpret_cast<QueueChunk<ChunkBase>*>(Ouro::ldg_cg(&chunk_ptr->next_))->count_) == static_cast<unsigned long long>(num_spots_))
		{
			chunk_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(Ouro::ldg_cg(&chunk_ptr->next_));
		}
		else
			break;
	}

	// We return the number of successfull "pointer shifts", we can use this number to actually free up some chunks later on
	return ret_val;
}

// ##############################################################################################################################################
//
template <typename ChunkBase>
template <typename MemoryManagerType>
__forceinline__ __device__ void QueueChunk<ChunkBase>::setOldPointer(MemoryManagerType* memory_manager, QueueChunk<ChunkBase>** queue_old_ptr, unsigned int* old_count, unsigned int free_count)
{
	using ChunkType = typename MemoryManagerType::ChunkType;
	// Read current old count
	auto current_old_count = Ouro::ldg_cg(old_count);

	// This branch is only taken about "LARGEST_OLD_COUNT_VALUE" times, if old_count is larger we already know how much to free up
	if(current_old_count < LARGEST_OLD_COUNT_VALUE)
	{
		// Increase the old_count by how much we want to free up
		current_old_count = atomicAdd(old_count, free_count);
		
		// Check if the old count is smaller than threshold
		if(current_old_count < LARGEST_OLD_COUNT_VALUE)
		{
			// If old count is now larger, compute difference
			if((current_old_count + free_count) > LARGEST_OLD_COUNT_VALUE)
			{
				// We can free some old_chunks, cut old_count back
				free_count = (current_old_count + free_count) - LARGEST_OLD_COUNT_VALUE;
				atomicSub(old_count, free_count);
			}
			else
			{
				// If it is smaller, don't do anything yet!
				free_count = 0;
			}
		}
	}
	
	// Do NOT reorder here
	__threadfence_block();

	// Free up some old chunks
	if(free_count)
	{
		QueueChunk<ChunkBase>* current_old_ptr{reinterpret_cast<QueueChunk<ChunkBase>*>(Ouro::ldg_cg(reinterpret_cast<unsigned long long*>(queue_old_ptr)))};
		while(free_count > 0)
		{
			auto current_old_ptr_comp = current_old_ptr;
			// Try to set the current old pointer to the current old next pointer
			if((current_old_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(
				atomicCAS((reinterpret_cast<unsigned long long*>(queue_old_ptr)), reinterpret_cast<unsigned long long>(current_old_ptr), current_old_ptr->next_)))
				== current_old_ptr_comp)
			{
				--free_count;

				if(!FINAL_RELEASE && printDebug)
					printf("%d - %d Reuse index: %u \n", threadIdx.x, blockIdx.x, ChunkType::Base::getIndexFromPointer(memory_manager->d_data, current_old_ptr));
				memory_manager->template enqueueChunkForReuse<true>(ChunkType::Base::getIndexFromPointer(memory_manager->d_data, current_old_ptr));

				current_old_ptr = reinterpret_cast<QueueChunk<ChunkBase>*>(current_old_ptr->next_);
			}
		}
	}
}
