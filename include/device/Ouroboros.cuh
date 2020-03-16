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
	index_t next_free_chunk{ 0 };

	size_t maxChunks{ 0 };
	size_t allocationSize{ 0 };
	size_t adjacencysize{ 0 };
	size_t additionalSizeBeginning{0};
	size_t additionalSizeEnd{0};

	ChunkLocator chunk_locator;
};

struct OuroborosBase
{
	// Data
	memory_t* d_memory{ nullptr };
	memory_t* d_data{ nullptr };
	index_t* next_free_chunk{ nullptr };
	size_t maxChunks{ 0 };
	ChunkLocator* chunk_locator{nullptr};

	bool initialized{ false };

	// Re-Use Queue
	IndexQueue d_chunk_reuse_queue;

	// Sizes
	size_t allocationSize{ 0 };
	size_t mem_manager_size{ 0 };
	size_t chunkqueuesize{ 0 };
	size_t pagequeuessize{ 0 };
	size_t adjacencysize{ 0 };
	size_t additionalSizeBeginning{0};
	size_t additionalSizeEnd{0};

	// Some statistics in there
	Statistics stats;

	// Error Code
	Ouro::ErrorType error{Ouro::ErrorVal<Ouro::ErrorType, Ouro::ErrorCodes::NO_ERROR>::value};

	bool checkError() { return Ouro::ErrorVal<Ouro::ErrorType, Ouro::ErrorCodes::NO_ERROR>::checkError(error); }
};

template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
struct OuroborosChunks : OuroborosBase
{
	static constexpr bool ManageChunks{ true };

	static constexpr unsigned int SmallestPageSize_{SMALLEST_SIZE};
	static constexpr unsigned int NumberQueues_{NUMBER_QUEUES};
	static constexpr unsigned int LargestPageSize_{SmallestPageSize_ << (NumberQueues_ - 1)};
	static constexpr unsigned int ChunkSize_{ SmallestPageSize_ << (NumberQueues_ - 1) };
	static constexpr unsigned int ChunkAddFactor_{ChunkSize_ / CHUNK_BASE::size_};

	
	using ChunkBase = CHUNK_BASE;
	using ChunkType = ChunkIndexChunk<ChunkBase, ChunkSize_, SmallestPageSize_>;
	using QueueType = QUEUE_TYPE<ChunkType>;
	using QI = QueueIndex<SmallestPageSize_, ChunkSize_>;

	static constexpr size_t memory_manager_size_() { return Ouro::alignment<uint64_t>(sizeof(OuroborosChunks)); };
	static constexpr size_t chunk_queue_size_{Ouro::alignment<uint64_t>(chunk_queue_size * sizeof(index_t))};
	static constexpr size_t page_queue_size_{Ouro::alignment<uint64_t>(QueueType::size_ * sizeof(MemoryIndex))};

	QueueType d_storage_reuse_queue[NumberQueues_];

	static constexpr size_t memoryManagerSize() { return chunk_queue_size_ + (page_queue_size_ * NumberQueues_); }

	void initialize(size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

	void initializeNew(memory_t** d_data_end);

	void reinitialize(float overallocation_factor);

	__forceinline__ __device__ void* allocPage(size_t size);

	__forceinline__ __device__ void freePage(MemoryIndex index);

	__forceinline__ __device__ void initializeQueues();

	__forceinline__ __device__ void printFreeResources();

	// #################################################################################################
	// Functionality
	template <bool QUEUECHUNK = false>
	__forceinline__ __device__ bool allocateChunk(index_t& chunk_index)
	{
		#ifdef __CUDA_ARCH__

		if(statistics_enabled)
			atomicAdd(&stats.chunkAllocationCount, 1);
		if(d_chunk_reuse_queue.dequeue(chunk_index))
			return true;
		chunk_index = atomicAdd(next_free_chunk, (QUEUECHUNK ? 1 : ChunkAddFactor_));
		chunk_locator->initChunkIndex(chunk_index);
		return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

		#else

		chunk_index = *next_free_chunk;
		*next_free_chunk += (QUEUECHUNK ? 1 : ChunkAddFactor_);
		return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

		#endif
	}

	void printQueueStatistics()
	{
		printf("%sPage Queue Fill Rate\n%s", break_line_purple_s, break_line);
		for(auto i = 0; i < NumberQueues_; ++i)
		{
			Ouro::printProgressBar(d_storage_reuse_queue[i].getCount() / static_cast<double>(d_storage_reuse_queue[i].size_));
			Ouro::printProgressBarEnd();
		}
		printf("%s", break_line_purple);
	}
};

template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
struct OuroborosPages : OuroborosBase
{
	static constexpr bool ManageChunks{ false };

	static constexpr unsigned int SmallestPageSize_{SMALLEST_SIZE};
	static constexpr unsigned int NumberQueues_{NUMBER_QUEUES};
	static constexpr unsigned int LargestPageSize_{SmallestPageSize_ << (NumberQueues_ - 1)};
	static constexpr unsigned int ChunkSize_{ SmallestPageSize_ << (NumberQueues_ - 1) };
	static constexpr unsigned int ChunkAddFactor_{ChunkSize_ / CHUNK_BASE::size_};

	using ChunkBase = CHUNK_BASE;
	using ChunkType = PageChunk<ChunkBase, ChunkSize_>;
	using QueueType = QUEUE_TYPE<ChunkType>;
	using QI = QueueIndex<SmallestPageSize_, ChunkSize_>;

	static constexpr size_t memory_manager_size_() { return Ouro::alignment<uint64_t>(sizeof(OuroborosPages)); };
	static constexpr size_t chunk_queue_size_{Ouro::alignment<uint64_t>(chunk_queue_size * sizeof(index_t), ChunkSize_)};
	static constexpr size_t page_queue_size_{Ouro::alignment<uint64_t>(QueueType::size_ * sizeof(MemoryIndex), ChunkSize_)};

	QueueType d_storage_reuse_queue[NumberQueues_];

	static constexpr size_t memoryManagerSize() { return chunk_queue_size_ + (page_queue_size_ * NumberQueues_); }

	void initialize(size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

	void initializeNew(memory_t** d_data_end);

	void reinitialize(float overallocation_factor);

    __forceinline__ __device__ void* allocPage(size_t size);

	 __forceinline__ __device__ void freePage(MemoryIndex index);

	// #################################################################################################
	// Functionality
	template <bool QUEUECHUNK = false>
	__forceinline__ __device__ bool allocateChunk(index_t& chunk_index)
	{
		#ifdef __CUDA_ARCH__

		if(statistics_enabled)
			atomicAdd(&stats.chunkAllocationCount, 1);
		if(d_chunk_reuse_queue.dequeue(chunk_index))
		{
			return true;
		}
		chunk_index = atomicAdd(next_free_chunk, (QUEUECHUNK ? 1 : ChunkAddFactor_));
		chunk_locator->initChunkIndex(chunk_index);
		return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

		#else

		chunk_index = *next_free_chunk;
		*next_free_chunk += (QUEUECHUNK ? 1 : ChunkAddFactor_);
		return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

		#endif
	}

	__forceinline__ __device__ void initializeQueues();

	void printQueueStatistics()
	{
		printf("%sPage Queue Fill Rate\n%s", break_line_purple_s, break_line);
		for(auto i = 0; i < NumberQueues_; ++i)
		{
			Ouro::printProgressBar(d_storage_reuse_queue[i].getCount() / static_cast<double>(d_storage_reuse_queue[i].size_));
			Ouro::printProgressBarEnd();
		}
		printf("%s", break_line_purple);
	}

	__forceinline__ __device__ void printFreeResources();
};

template<class... OUROBOROSES>
struct Ouroboros;

template<class OUROBOROS, class... OUROBOROSES>
struct Ouroboros<OUROBOROS, OUROBOROSES...>
{
	using ConcreteOuroboros = OUROBOROS;
	using Next = Ouroboros<OUROBOROSES...>;
	using ChunkBase = typename OUROBOROS::ChunkBase;
	using ChunkType = typename OUROBOROS::ChunkType;
	using MyType = Ouroboros<OUROBOROS, OUROBOROSES...>;
	using QI = typename ConcreteOuroboros::QI;

	static constexpr size_t size_() { return Ouro::alignment<size_t>(sizeof(Ouroboros<OUROBOROS, OUROBOROSES...>), ChunkBase::size_); };

	Memory memory;
	ConcreteOuroboros memory_manager;
	Next next_memory_manager;

	bool initialized{false};
	Statistics stats;

	// -----------------------------------------------------------------------------------------------------------
	// Public Interface

	~Ouroboros();

	void initialize(size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

	void reinitialize(float overallocation_factor);

	__forceinline__ __device__ void* malloc(size_t size);

	__forceinline__ __device__ void free(void* ptr);

	__forceinline__ __device__ void freePageRecursive(unsigned int page_size, MemoryIndex index);

	__forceinline__ __device__ void enqueueInitialChunk(index_t queue_index, index_t chunk_index, int available_pages, index_t pages_per_chunk)
	{
		// TODO: This should later relay to the correct mem man
		printf("This should relay to the correct memory manager!\n");
		__trap();

		memory_manager.d_storage_reuse_queue[queue_index].enqueueInitialChunk(&memory_manager, chunk_index, available_pages, pages_per_chunk);
	}

	MyType* getDeviceMemoryManager(){return reinterpret_cast<MyType*>(memory.d_memory);}

	// -----------------------------------------------------------------------------------------------------------
	// Private Interface

	static constexpr int totalNumberQueues()
	{
		return ((ConcreteOuroboros::QueueType::virtualized) ? ConcreteOuroboros::NumberQueues_ : 0) 
		+ Next::totalNumberQueues();
	}

	size_t totalMemoryManagerSize()
	{
		return memory_manager.memoryManagerSize() + next_memory_manager.totalMemoryManagerSize();
	}

	void initMemoryManagers()
	{
		init(&memory);
	}

	void init(Memory* memory)
	{
		memory_manager.initializeNew(&(memory->d_data_end));
		next_memory_manager.init(memory);
	}

	__forceinline__ __device__ void setMemoryPointer()
	{
		setMemory(&memory);
	}
	
	__forceinline__ __device__ void setMemory(Memory* memory)
	{
		memory_manager.d_memory = memory->d_memory;
		memory_manager.d_data = memory->d_data;
		memory_manager.next_free_chunk = &(memory->next_free_chunk);
		memory_manager.maxChunks = memory->maxChunks;
		memory_manager.chunk_locator = &(memory->chunk_locator);
		next_memory_manager.setMemory(memory);
	}

	__forceinline__ __device__ void initQueues();

	void printFreeResources();

	__forceinline__ __device__ void d_printResources();

	__forceinline__ __device__ bool validOuroborosPointer(void* ptr)
	{
		if(reinterpret_cast<unsigned long long>(ptr) > reinterpret_cast<unsigned long long>(memory.d_memory) 
		&& reinterpret_cast<unsigned long long>(ptr) < (reinterpret_cast<unsigned long long>(memory.d_memory) + memory.allocationSize))
			return true;
		return false;
	}

	// Temporary Memory Allocator
	struct HeapAllocator
	{
		HeapAllocator() : d_memory{nullptr} {}
		HeapAllocator(memory_t* ptr) : d_memory{ptr} {}
		memory_t* d_memory{nullptr};
		size_t allocated_size{0};

		template <typename DataType>
		DataType* getMemoryInBytes(size_t num_Bytes)
		{
			d_memory -= num_Bytes;
			allocated_size += num_Bytes;
			return reinterpret_cast<DataType*>(d_memory);
		}

		template <typename DataType>
		DataType* getMemoryInItems(size_t num_Items)
		{
			d_memory -= num_Items * sizeof(DataType);
			allocated_size += num_Items * sizeof(DataType);
			return reinterpret_cast<DataType*>(d_memory);
		}
	};

	HeapAllocator createHeapAllocator(unsigned int offset_in_bytes = 0)
	{
		printf("Not implemented correctly yet!\n");
		exit(-1);
		return HeapAllocator(memory.d_data_end);
	}
};

template <>
struct Ouroboros<>
{
	void init(Memory* memory) {}
	size_t totalMemoryManagerSize() {return 0ULL;}

	__forceinline__ __device__ void* malloc(size_t size)
	{
	#ifdef __CUDA_ARCH__
		return ::malloc(AllocationHelper::getNextPow2(size));
	#else
		return nullptr;
	#endif
	}

	__forceinline__ __device__ void freePageRecursive(unsigned int page_size, MemoryIndex index)
	{
		printf("Spilled into empty Ouroboros, this should not happend\n");
		__trap();
	}

	__forceinline__ __device__ void setMemory(Memory* memory){}
	__forceinline__ __device__ void initQueues() {}
	__forceinline__ __device__ void d_printResources(){}
	void printFreeResources(){}
	static constexpr int totalNumberQueues(){return 0;}
};

template <typename MemoryManagerType>
void updateMemoryManagerDevice(MemoryManagerType& memory_manager);
template <typename MemoryManagerType>
void updateMemoryManagerHost(MemoryManagerType& memory_manager);