#include <iostream>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

#define TEST_MULTI

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType* mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm->malloc(allocation_size));
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	if(threadIdx.x == 0 && blockIdx.x == 0)
		printf("Test Read!\n");
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d - %d | We got a wrong value here! %d vs %d\n", threadIdx.x, blockIdx.x, ptr[i], tid);
			return;
		}
	}
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType* mm, int** verification_ptr, int num_allocations)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	mm->free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
	std::cout << "Usage: num_allocations allocation_size_in_bytes\n";
	int num_allocations{10000};
	int allocation_size_byte{16};
	int num_iterations {10};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
		}
	}
	allocation_size_byte = Ouro::alignment(allocation_size_byte, sizeof(int));
	std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << " | Iterations: " << num_iterations << std::endl;

	#ifdef TEST_PAGES

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing page-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVAPQ;
	#else
	using MemoryManagerType = MultiOuroVAPQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing page-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLPQ;
	#else
	using MemoryManagerType = MultiOuroVLPQ;
	#endif
	#else
	std::cout << "Testing page-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroPQ;
	#else
	using MemoryManagerType = MultiOuroPQ;
	#endif
	#endif

	#elif TEST_CHUNKS

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing chunk-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVACQ;
	#else
	using MemoryManagerType = MultiOuroVACQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing chunk-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLCQ;
	#else
	using MemoryManagerType = MultiOuroVLCQ;
	#endif
	#else
	std::cout << "Testing chunk-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroCQ;
	#else
	using MemoryManagerType = MultiOuroCQ;
	#endif
	#endif

	#endif

	size_t instantitation_size = 8192ULL * 1024ULL * 1024ULL;
	MemoryManagerType memory_manager;
	memory_manager.initialize(instantitation_size);

	int** d_memory{nullptr};
	HANDLE_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	int blockSize {256};
	int gridSize {Ouro::divup(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
	cudaEvent_t start, end;
	for(auto i = 0; i < num_iterations; ++i)
	{
		start_clock(start, end);
		d_testAllocation <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations, allocation_size_byte);
		timing_allocation += end_clock(start, end);

		HANDLE_ERROR(cudaDeviceSynchronize());

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		HANDLE_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);

		HANDLE_ERROR(cudaDeviceSynchronize());

		start_clock(start, end);
		d_testFree <MemoryManagerType> <<<gridSize, blockSize>>>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations);
		timing_free += end_clock(start, end);

		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
	std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

	std::cout << "Testcase DONE!\n";
	
	return 0;
}