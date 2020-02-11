#include <iostream>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

template <typename MemoryManagerType>
__global__ void d_testKernel(MemoryManagerType* mm)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= 16)
		return;

	int* test_array = reinterpret_cast<int*>(mm->malloc(sizeof(int) * 16));

	for(int i = 0; i < 16 ; ++i)
	{
		test_array[i] = i + tid;
	}

	mm->free(test_array);
}

int main()
{
	#ifdef TEST_PAGES

	#ifdef TEST_VIRTUALARRAY
	using MemoryManagerType = OuroVAPQ;
	std::cout << "Testing page-based memory manager - Virtualized Array!\n";
	#elif TEST_VIRTUALLIST
	std::cout << "Testing page-based memory manager - Virtualized List!\n";
	using MemoryManagerType = OuroVLPQ;
	#else
	std::cout << "Testing page-based memory manager - Standard!\n";
	using MemoryManagerType = OuroPQ;
	#endif

	#elif TEST_CHUNKS

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing chunk-based memory manager - Virtualized Array!\n";
	using MemoryManagerType = OuroVACQ;
	#elif TEST_VIRTUALLIST
	std::cout << "Testing chunk-based memory manager - Virtualized List!\n";
	using MemoryManagerType = OuroVLCQ;
	#else
	std::cout << "Testing chunk-based memory manager - Standard!\n";
	using MemoryManagerType = OuroCQ;
	#endif

	#endif

	MemoryManagerType memory_manager;
	memory_manager.initialize();

	d_testKernel <MemoryManagerType> <<<1, 32>>>(memory_manager.getDeviceMemoryManager());

	cudaDeviceSynchronize();
	
	return 0;
}