cmake_minimum_required(VERSION 3.16)
project(ouroboros LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

option(TEST_P "Test Page-Based Standard" True)
option(TEST_VAP "Test Page-Based VA" True)
option(TEST_VLP "Test Page-Based VL" True)
option(TEST_C "Test Chunk-Based Standard" True)
option(TEST_VAC "Test Chunk-Based VA" True)
option(TEST_VLC "Test Chunk-Based VL" True)

##########################################################################
# Executables
##########################################################################

SET(OUROBOROSHEADERS
	# General Stuff
	include/Definitions.h
	include/Utility.h
	include/Utility.cuh
	include/PerformanceMeasure.cuh
	include/Parameters.h
	include/device/CudaUniquePtr.cuh
	include/helper/json.h
	include/device/Helper.cuh

	# Access Primitive
	include/device/BulkSemaphore.cuh
	include/device/BulkSemaphore_impl.cuh

	# Memory
	include/device/MemoryIndex.cuh
	include/device/MemoryQueries.cuh

	# Chunk Stuff
	include/device/Chunk.cuh
	include/device/ChunkIndexChunk.cuh
	include/device/ChunkAccess.cuh
	include/device/ChunkAccess_impl.cuh
	include/device/PageIndexChunk.cuh
	include/device/queues/QueueChunk.cuh
	include/device/queues/QueueChunk_impl.cuh

	# Queues
	include/device/queues/Queue.h
	include/device/queues/Queue_impl.cuh
	include/device/queues/Queues.cuh
	include/device/queues/Queues_impl.cuh
	include/device/queues/page/PageQueue.cuh
	include/device/queues/page/PageQueue_impl.cuh
	include/device/queues/page/PageQueueVA.cuh
	include/device/queues/page/PageQueueVA_impl.cuh
	include/device/queues/page/PageQueueVL.cuh
	include/device/queues/page/PageQueueVL_impl.cuh
	include/device/queues/chunk/ChunkQueue.cuh
	include/device/queues/chunk/ChunkQueue_impl.cuh
	include/device/queues/chunk/ChunkQueueVA.cuh
	include/device/queues/chunk/ChunkQueueVA_impl.cuh
	include/device/queues/chunk/ChunkQueueVL.cuh
	include/device/queues/chunk/ChunkQueueVL_impl.cuh

	# Ouroboros
	include/device/Ouroboros.cuh
	include/device/Ouroboros_impl.cuh
	include/device/MemoryInitialization.cuh
)

if(TEST_P)
add_executable(main_p
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(main_p PUBLIC TEST_PAGES)
target_include_directories(main_p PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(TEST_C)
add_executable(main_c
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(main_c PUBLIC TEST_CHUNKS)
target_include_directories(main_c PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(TEST_VAP)
add_executable(va_main_p
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(va_main_p PUBLIC TEST_PAGES TEST_VIRTUALARRAY)
target_include_directories(va_main_p PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(TEST_VAC)
add_executable(va_main_c
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(va_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALARRAY)
target_include_directories(va_main_c PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(TEST_VLP)
add_executable(vl_main_p
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(vl_main_p PUBLIC TEST_PAGES TEST_VIRTUALLIST)
target_include_directories(vl_main_p PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(TEST_VLC)
add_executable(vl_main_c
	src/main.cu
	${OUROBOROSHEADERS}
)
target_compile_definitions(vl_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALLIST)
target_include_directories(vl_main_c PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()