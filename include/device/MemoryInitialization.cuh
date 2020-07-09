#pragma once

#include "Ouroboros_impl.cuh"

// ##############################################################################################################################################
//
__global__ void printCompute()
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= 1)
		return;
	#if (__CUDA_ARCH__ >= 700)
			printf("ASYNC - COMPUTE MODE");
		#else
			printf("SYNC - COMPUTE MODE");
		#endif
}

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
__global__ void d_cleanChunks(MemoryManagerType* memory_manager, unsigned int offset)
{
	using ChunkType = typename MemoryManagerType::ChunkBase;
	__shared__ index_t* chunk_data;

	if(threadIdx.x == 0)
	{
		 chunk_data = reinterpret_cast<index_t*>(ChunkType::getMemoryAccess(memory_manager->memory.d_data, blockIdx.x + offset));
	}
	
	__syncthreads();

	for (int i = threadIdx.x; i < (MemoryManagerType::ChunkBase::size_ + MemoryManagerType::ChunkBase::meta_data_size_); i += blockDim.x)
	{
		chunk_data[i] = DeletionMarker<index_t>::val;
	}
}

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
void initNew(MemoryManagerType& memory_manager, memory_t** d_data_end)
{
	// Place Chunk Queue
	*d_data_end -= MemoryManagerType::chunk_queue_size_;
	memory_manager.d_chunk_reuse_queue.queue_ = reinterpret_cast<decltype(memory_manager.d_chunk_reuse_queue.queue_)>(*d_data_end);
	memory_manager.d_chunk_reuse_queue.size_ = chunk_queue_size;

	// Place Page Queues
	for (auto i = 0; i < MemoryManagerType::NumberQueues_; ++i)
	{
		*d_data_end -= MemoryManagerType::page_queue_size_;
		memory_manager.d_storage_reuse_queue[i].queue_ = reinterpret_cast<index_t*>(*d_data_end);
		memory_manager.d_storage_reuse_queue[i].queue_index_ = i;
		memory_manager.d_storage_reuse_queue[i].page_size_ = MemoryManagerType::SmallestPageSize_ << i;
	}
}

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
void OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::initializeNew(memory_t** d_data_end)
{
	initNew<OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(*this, d_data_end);
}

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
void OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::initializeNew(memory_t** d_data_end)
{
	initNew<OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(*this, d_data_end);
}

// ##############################################################################################################################################
//
template <typename OUROBOROS>
__global__ void d_setupMemoryPointers(OUROBOROS* ouroboros)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid != 0)
		return;

	// Template-recursive give all memory managers the same pointer
	ouroboros->setMemoryPointer();
}

// ##############################################################################################################################################
//
template <typename OUROBOROS>
__global__ void d_initializeOuroborosQueues(OUROBOROS* ouroboros)
{
	// Template-recursive to initialize queues
	ouroboros->memory.chunk_locator.init(ouroboros->memory.maxChunks);
	IndexQueue* d_base_chunk_reuse{nullptr};
	ouroboros->initQueues(d_base_chunk_reuse);
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
__forceinline__ __device__ void Ouroboros<OUROBOROS, OUROBOROSES...>::initQueues(IndexQueue* d_base_chunk_reuse)
{
	// --------------------------------------------------------
	// Init queues
	memory_manager.d_chunk_reuse_queue.init();
	if(d_base_chunk_reuse == nullptr)
		d_base_chunk_reuse = &(memory_manager.d_chunk_reuse_queue);
	memory_manager.d_base_chunk_reuse_queue = d_base_chunk_reuse;
	#pragma unroll
	for (auto i = 0; i < ConcreteOuroboros::NumberQueues_; ++i)
	{
		memory_manager.d_storage_reuse_queue[i].init(&memory_manager);
	}

	// Init next queues
	next_memory_manager.initQueues(d_base_chunk_reuse);
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
void Ouroboros<OUROBOROS, OUROBOROSES...>::initialize(size_t instantiation_size, size_t additionalSizeBeginning, size_t additionalSizeEnd)
{
	// Initialize memory, then call initialize on all instances
	if (initialized)
		return;
	
	if(!cuda_initialized)
	{
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_heap_size);
		cuda_initialized = true;
	}
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	if(printDebug)
		printf("Heap Size: ~%llu MB\n", size / (1024 * 1024));

	if(printDebug)
	{
		printf("%s##\n####\n##\n---", break_line_green_s);
		printCompute<<<1, 32>>>();
		cudaDeviceSynchronize();
		#ifdef TEST_VIRTUALIZED
		printf(" - VIRTUALIZED ARRAY-BASED");
		#elif TEST_VIRTUALIZED_LINKED
		printf(" - VIRTUALIZED LINKED-LIST-BASED");
		#else
		printf(" - STANDARD");
		#endif
		printf("  ---\n##\n####\n##\n%s", break_line_green_e);
	}

	// Get total size from all Memory Managers
	auto total_memory_manager_size = totalMemoryManagerSize();

	// Align both the required size and total size to the chunk base size
	auto total_required_size = Ouro::alignment<size_t>(size_() + total_memory_manager_size + additionalSizeBeginning + additionalSizeEnd, ChunkBase::size());
	auto difference = Ouro::alignment<size_t>(instantiation_size, ChunkBase::size()) - total_required_size;
	memory.maxChunks = difference / ChunkBase::size();
	memory.adjacencysize = Ouro::alignment<uint64_t>(memory.maxChunks * ChunkBase::size());
	size_t chunk_locator_size = Ouro::alignment<size_t>(ChunkLocator::size(memory.maxChunks), ChunkBase::size());
	memory.allocationSize = Ouro::alignment<uint64_t>(total_required_size + memory.adjacencysize + chunk_locator_size, ChunkBase::size());
	memory.additionalSizeBeginning = additionalSizeBeginning;
	memory.additionalSizeEnd = additionalSizeEnd;

	// Allocate memory
	if (!memory.d_memory)
		cudaMalloc(reinterpret_cast<void**>(&memory.d_memory), memory.allocationSize);

	memory.d_data = memory.d_memory + size_();
	memory.d_data_end = memory.d_memory + memory.allocationSize;

	// Put Memory Manager on Device
	updateMemoryManagerDevice(*this);

	d_setupMemoryPointers<MyType><<<1, 1>>>(reinterpret_cast<MyType*>(memory.d_memory));

	HANDLE_ERROR(cudaDeviceSynchronize());

	// Update pointers on host
	updateMemoryManagerHost(*this);

	// Place chunk locator
	memory.d_data_end -= chunk_locator_size;
	memory.chunk_locator.d_chunk_flags = reinterpret_cast<int*>(memory.d_data_end);

	// Lets distribute this pointer to the memory managers
	initMemoryManagers();

	// Lets update the device again to that all the info is there as well
	updateMemoryManagerDevice(*this);

	int block_size = 256;
	int grid_size = memory.maxChunks;
	if(totalNumberVirtualQueues())
	{
		// Clean all chunks
		d_cleanChunks<MyType> << <grid_size, block_size >> > (reinterpret_cast<MyType*>(memory.d_memory), 0);
	}

	HANDLE_ERROR(cudaDeviceSynchronize());

	block_size = 256;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, d_initializeOuroborosQueues<MyType>, block_size, 0U);
	int num_sm_per_device{0};
	cudaDeviceGetAttribute(&num_sm_per_device, cudaDevAttrMultiProcessorCount, 0);
	grid_size *= num_sm_per_device;
	d_initializeOuroborosQueues<MyType> << <grid_size, block_size >> > (reinterpret_cast<MyType*>(memory.d_memory));

	HANDLE_ERROR(cudaDeviceSynchronize());

	updateMemoryManagerHost(*this);

	initialized = true;

	HANDLE_ERROR(cudaDeviceSynchronize());
}