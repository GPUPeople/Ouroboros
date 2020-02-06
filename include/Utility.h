#pragma once

#include "Definitions.h"

namespace Ouro
{
	template <typename T>
	static constexpr bool isPowerOfTwo(T n) {return (n & (n - 1)) == 0;}

	template<typename T>
	__host__ __device__ __forceinline__ T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}

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
		return divup<T>(size, alignment) * alignment;
	}

	// ##############################################################################################################################################
	//
	template<typename T>
	__host__ __device__ __forceinline__ size_t sizeofInBits()
	{
		return sizeof(T) * BYTE_SIZE;
	}

	template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

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

	template <typename T, typename SizeType>
	static constexpr __forceinline__ __device__ T modPower2(T value, SizeType size)
	{
		return value & (size - 1);
	}

	template <unsigned int size>
	static constexpr __forceinline__ __device__ unsigned int modPower2(const unsigned int value)
	{
		static_assert(isPowerOfTwo(size), "ModPower2 used with non-power of 2");
		return value & (size - 1);
	}
}