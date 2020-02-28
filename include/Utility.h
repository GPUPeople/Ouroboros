#pragma once

#include <string>

#include "Definitions.h"
#include "Parameters.h"

// ##############################################################################################################################################
//
static inline void HandleError(cudaError_t err,
	const char* string,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s%s\n%s", break_line_red_s, string, break_line_red_e);
		printf("%s in \n\n%s at line %d\n", cudaGetErrorString(err),
			file, line);
		printf("%s", break_line_red);
		exit(EXIT_FAILURE);
	}
}

// ##############################################################################################################################################
//
static inline void HandleError(const char *file,
	int line) {
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, "", __FILE__, __LINE__ ))
#define HANDLE_ERROR_S( err , string) (HandleError( err, string, __FILE__, __LINE__ ))

// ##############################################################################################################################################
	//
static inline void DEBUG_checkKernelError(const char* message = nullptr)
{
	if (debug_enabled)
	{
		HANDLE_ERROR(cudaPeekAtLastError());
		HANDLE_ERROR(cudaDeviceSynchronize());
		if (printDebug && message)
			printf("%s\n", message);
	}
}

void queryAndPrintDeviceProperties();


// ##############################################################################################################################################
//
void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&end));
	HANDLE_ERROR(cudaEventRecord(start, 0));
}

// ##############################################################################################################################################
//
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, 0));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(end));

	// Returns ms
	return time;
}

namespace Ouro
{
	// ##############################################################################################################################################
	//
	static constexpr __forceinline__ __device__ unsigned long long create2Complement(unsigned long long value)
	{
		return ~(value) + 1ULL;
	}

	// ##############################################################################################################################################
	//
	template <typename T>
	static constexpr bool isPowerOfTwo(T n) 
	{
		return (n & (n - 1)) == 0;
	}

	// ##############################################################################################################################################
	//
	template<typename T>
	__host__ __device__ __forceinline__ T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}

	// ##############################################################################################################################################
	//
	template<typename T, typename O>
	constexpr __host__ __device__ __forceinline__ T divup(T a, O b)
	{
		return (a + b - 1) / b;
	}

	// ##############################################################################################################################################
	//
	template<typename T>
	constexpr __host__ __device__ __forceinline__ T alignment(const T size, size_t alignment = CACHELINE_SIZE)
	{
		return divup<T, size_t>(size, alignment) * alignment;
	}

	// ##############################################################################################################################################
	//
	template<typename T>
	constexpr __host__ __device__ __forceinline__ size_t sizeofInBits()
	{
		return sizeof(T) * BYTE_SIZE;
	}

	// ##############################################################################################################################################
	//
	static constexpr int countBitShift(unsigned int x)
	{
		if (x == 0) return 0;
		int n = 0;
		if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
		if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
		if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
		if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
		if (x <= 0x7FFFFFFF) { n = n + 1; x = x << 1; }
		return 31 - n;
	}

	// ##############################################################################################################################################
	//
	template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};

	// ##############################################################################################################################################
	//
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

	// ##############################################################################################################################################
	//
	static constexpr int cntlz(unsigned int x)
	{
		if (x == 0) return 32;
		int n = 0;
		if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
		if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
		if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
		if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
		if (x <= 0x7FFFFFFF) { n = n + 1; x = x << 1; }
		return n;
	}

	// ##############################################################################################################################################
	//
	static inline void printTestcaseSeparator(const std::string& header)
	{
		printf("%s", break_line_purple_s);
		printf("#%105s\n", "#");
		printf("###%103s\n", "###");
		printf("#####%101s\n", "#####");
		printf("#######%99s\n", "#######");
		printf("#########%55s%42s\n", header.c_str(), "#########");
		printf("#######%99s\n", "#######");
		printf("#####%101s\n", "#####");
		printf("###%103s\n", "###");
		printf("#%105s\n", "#");
		printf("%s", break_line_purple_e);
	}

	static constexpr char PBSTR[] = "##############################################################################################################";
	static constexpr int PBWIDTH = 99;

	// ##############################################################################################################################################
	//
	static inline void printProgressBar(const double percentage)
	{
		auto val = static_cast<int>(percentage * 100);
		auto lpad = static_cast<int>(percentage * PBWIDTH);
		auto rpad = PBWIDTH - lpad;
	#ifdef WIN32
		printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	#else
		printf("\r\033[0;35m%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	#endif
		fflush(stdout);
	}

	// ##############################################################################################################################################
	//
	static inline void printProgressBarEnd()
	{
	#ifdef WIN32
		printf("\n");
	#else
		printf("\033[0m\n");
	#endif
		fflush(stdout);
	}

	// ##############################################################################################################################################
	//
	template <typename Data>
	void updateDataHost(Data& data)
	{
		HANDLE_ERROR(cudaMemcpy(&data,
			data.d_memory,
			sizeof(Data),
			cudaMemcpyDeviceToHost));
	}

	// ##############################################################################################################################################
	//
	template <typename Data>
	void updateDataDevice(Data& data)
	{
		HANDLE_ERROR(cudaMemcpy(data.d_memory,
			&data,
			sizeof(Data),
			cudaMemcpyHostToDevice));
	}

	// ##############################################################################################################################################
	//
	template <typename T, typename SizeType>
	static constexpr __forceinline__ __device__ T modPower2(T value, SizeType size)
	{
		return value & (size - 1);
	}

	// ##############################################################################################################################################
	//
	template <unsigned int size>
	static constexpr __forceinline__ __device__ unsigned int modPower2(const unsigned int value)
	{
		static_assert(isPowerOfTwo(size), "ModPower2 used with non-power of 2");
		return value & (size - 1);
	}

	// ##############################################################################################################################################
	// Error Codes
	using ErrorType = unsigned int;
	enum class ErrorCodes
	{
		NO_ERROR,
		OUT_OF_CUDA_MEMORY,
		OUT_OF_CHUNK_MEMORY,
		CHUNK_ENQUEUE_ERROR
	};

	// ##############################################################################################################################################
	//
	template <typename DataType, ErrorCodes Error>
	struct ErrorVal;

	// ##############################################################################################################################################
	//
	template <typename DataType>
	struct ErrorVal<DataType, ErrorCodes::NO_ERROR>
	{
		static constexpr DataType value{ 0 };

		static constexpr __forceinline__ __device__ void setError(ErrorType& error)
		{
	#ifdef __CUDA_ARCH__
			atomicOr(&error, value);
	#else
			error |= value;
	#endif
		}

		static constexpr __forceinline__ __device__ bool checkError(ErrorType& error)
		{
			return error != 0;
		}

		static constexpr __forceinline__ __device__ __host__ void print()
		{
			printf("No Error\n");
		}
	};

	// ##############################################################################################################################################
	//
	template <typename DataType>
	struct ErrorVal<DataType, ErrorCodes::OUT_OF_CUDA_MEMORY>
	{
		static constexpr DataType value{ 1 << 0 };

		static constexpr __forceinline__ __device__ void setError(ErrorType& error)
		{
	#ifdef __CUDA_ARCH__
			atomicOr(&error, value);
	#else
			error |= value;
	#endif
		}

		static constexpr __forceinline__ __device__ bool checkError(ErrorType& error)
		{
			return error & value;
		}

		static constexpr __forceinline__ __device__ __host__ void print()
		{
			printf("Out of CUDA Memory Error\n");
		}
	};

	// ##############################################################################################################################################
	//
	template <typename DataType>
	struct ErrorVal<DataType, ErrorCodes::OUT_OF_CHUNK_MEMORY>
	{
		static constexpr DataType value{ 1 << 1 };

		static constexpr __forceinline__ __device__ void setError(ErrorType& error)
		{
	#ifdef __CUDA_ARCH__
			atomicOr(&error, value);
	#else
			error |= value;
	#endif
		}

		static constexpr __forceinline__ __device__ bool checkError(ErrorType& error)
		{
			return error & value;
		}

		static constexpr __forceinline__ __device__ __host__ void print()
		{
			printf("Out of Chunk Memory Error\n");
		}
	};

	// ##############################################################################################################################################
	//
	template <typename DataType>
	struct ErrorVal<DataType, ErrorCodes::CHUNK_ENQUEUE_ERROR>
	{
		static constexpr DataType value{ 1 << 2 };

		static constexpr __forceinline__ __device__ void setError(ErrorType& error)
		{
	#ifdef __CUDA_ARCH__
			atomicOr(&error, value);
	#else
			error |= value;
	#endif
		}

		static constexpr __forceinline__ __device__ bool checkError(ErrorType& error)
		{
			return error & value;
		}

		static constexpr __forceinline__ __device__ __host__ void print()
		{
			printf("Chunk Enqueue Error\n");
		}
	};
}