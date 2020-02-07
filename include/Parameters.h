#pragma once

#include "Utility.h"

static constexpr bool FINAL_RELEASE{false};

// General Params
static constexpr int CACHELINE_SIZE{ 128 };
static constexpr int WARP_SIZE{ 32 };
static constexpr int UNLOCK{ 0 };
static constexpr int LOCK{ 1 };
static constexpr unsigned int SLEEP_TIME{ 10 };
// static constexpr unsigned long long ALLOCATION_SIZE{ 9ULL * 1024ULL * 1024ULL * 1024ULL };
static constexpr unsigned long long ALLOCATION_SIZE{ 2048ULL * 1024ULL * 1024ULL };

// Queue Params
static constexpr int NUM_QUEUES{ 10 }; // How many queues to instantiate per Ouroboros instance
static constexpr int chunk_queue_size{ 65536 }; // Size of chunk queue for re-use of chunks
static constexpr int page_queue_size{ 1024 * 1024 * 2 }; // Size of page queue
static constexpr int virtual_queue_size{ 1024 }; // Size of virtual page queue
static constexpr unsigned int LARGEST_OLD_COUNT_VALUE{10}; // How many old chunks to holds back before releasing them
static constexpr float LOWER_FILL_LEVEL_PERCENTAGE{0.2f}; // When to start releasing chunks from chunk queue

// Memory Params
static constexpr int SMALLEST_PAGE_SIZE{ 16}; // Smallest page size (16 Bytes)
static constexpr int CHUNK_SIZE{ SMALLEST_PAGE_SIZE << (NUM_QUEUES - 1) }; // Chunksize computed from smallest page size and number of queues
static constexpr int CHUNK_METADATA_SIZE{ 128 };
static constexpr int NUM_BITS_FOR_PAGE{ Ouro::countBitShift(CHUNK_SIZE / SMALLEST_PAGE_SIZE) }; // How many bits do we need for the page index in the combined index

// Print & Statistics
static constexpr bool turn_off_all_print_output{false};
static constexpr bool turn_off_all_metric{ turn_off_all_print_output || true };
static constexpr bool fragmentation_print{false};

static constexpr bool debug_enabled{ !turn_off_all_metric && true };
static constexpr bool printDebug{ debug_enabled && true };
static constexpr bool printDebugCUDA{ printDebug && false };

static constexpr bool statistics_enabled{ !turn_off_all_metric && true };
static constexpr bool printStats{ statistics_enabled && true };

static constexpr bool turnOnProgressBar{!turn_off_all_print_output && true};

// Heap Size
static constexpr size_t cuda_heap_size {50ULL * 1024ULL * 1024ULL};
static constexpr size_t cuda_mallocator_heap_size {2048ULL * 1024ULL * 1024ULL};
//static constexpr size_t cuda_mallocator_heap_size {50ULL * 1024ULL * 1024ULL};
static constexpr bool testCUDA{false};

// Performance
static constexpr bool showPerformanceOutput{!turn_off_all_print_output && true};
static constexpr bool showPerformanceOutputPerRound{showPerformanceOutput && true};

// Testing parameters
static constexpr bool preAllocateMemory{false};
