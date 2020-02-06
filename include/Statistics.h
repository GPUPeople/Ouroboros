#pragma once

#include "Definitions.h"

struct Statistics
{
	memory_t* d_memory;
	unsigned int availableChunks{ 0 };
	unsigned int cudaMallocCount{ 0 };
	unsigned int pageAllocCount{ 0 };
	unsigned int cudaFreeCount{ 0 };
	unsigned int pageFreeCount{ 0 };
	unsigned int chunkAllocationCount{ 0 };
	unsigned int chunkReuseCount{ 0 };
	unsigned int longest_adjacency{ 0 };
	unsigned int shortest_adjacency{ std::numeric_limits<unsigned int>::max() };
	unsigned int duplicates_detected{0};

    void printHeader(const char* header_title)
	{
        printf("%s"
               "%s\n"
               "%s", break_line_cyan_s, header_title, break_line_cyan_e);
	}
	void printFooter()
	{
        printf("%s", break_line_cyan);
	}

    void printAllocCount(const char* header_title)
	{
		printHeader(header_title);
		printf("%9u : Available Chunks\n", availableChunks);
		printf("%9u : Chunk Alloc Count\n", chunkAllocationCount);
		printf("%9u : Chunk Reuse Count\n", chunkReuseCount);
		printf("%9u : Page Alloc Count\n", pageAllocCount);
		printf("%9u : Page Free Count\n", pageFreeCount);
		printf("%9u : Cuda Malloc Count\n", cudaMallocCount);
		printf("%9u : Cuda Free Count\n", cudaFreeCount);
		printf("%9u : Longest Adjacency\n", longest_adjacency);
		printf("%9u : Shortest Adjacency\n", shortest_adjacency);
		printFooter();
	}

	template <typename EdgeDataType, typename MemoryManagerType>
    void printPageDistribution(const char* header_title, uint32_t* page_requirements, uint32_t* chunk_requirements, uint32_t overall_chunks, size_t chunk_size)
    {
        printHeader(header_title);
        for (auto i = 0; i < MemoryManagerType::NumberQueues_; ++i)
        {
            printf("Page Size: %6u Bytes | Pages/Chunk: %6u | Edges/Page: %6u | Pages: %8u | Chunks: %6u\n", (MemoryManagerType::SmallestPageSize_ * (1 << i)), (MemoryManagerType::ChunkSize_ / MemoryManagerType::SmallestPageSize_) >> i, (MemoryManagerType::SmallestPageSize_ * (1 << i)) / static_cast<unsigned int>(sizeof(EdgeDataType)), page_requirements[i], chunk_requirements[i]);
        }
        printf("----------------------------------\n");
        printf("Overall Chunks: %u\n", overall_chunks);
        printf("----------------------------------\n");
        printFooter();
    }
};

// #################################################################
//
inline std::ostream& writeGPUInfo(std::ostream& file)
{
	int cudaDevice;
	cudaGetDevice(&cudaDevice);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cudaDevice);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	file << "name;cc;num_multiprocessors;warp_size;max_threads_per_mp;regs_per_mp;shared_memory_per_mp;"
	"total_constant_memory;total_global_memory;clock_rate;max_threads_per_block;max_regs_per_block;max_shared_memory_per_block\n"
		<< prop.name << ';'
		<< prop.major << '.'
		<< prop.minor << ';'
		<< prop.multiProcessorCount << ';'
		<< prop.warpSize<< ';'
		<< prop.maxThreadsPerMultiProcessor << ';'
		<< prop.regsPerMultiprocessor << ';'
		<< prop.sharedMemPerMultiprocessor << ';'
		<< prop.totalConstMem << ';'
		<< prop.totalGlobalMem << ';'
		<< prop.clockRate * 1000 << ';'
		<< prop.maxThreadsPerBlock << ';'
		<< prop.regsPerBlock << ';'
		<< prop.sharedMemPerBlock
		<< std::endl;
	return file;
}