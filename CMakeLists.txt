cmake_minimum_required(VERSION 3.16)
project(ouroboros LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

##########################################################################
# Include Directories
include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

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
	include/device/queues/chunk/ChunkQueue.cuh
	include/device/queues/chunk/ChunkQueue_impl.cuh

	# Ouroboros
	include/device/Ouroboros.cuh
)

add_executable(main
	src/main.cu
	${OUROBOROSHEADERS}
)