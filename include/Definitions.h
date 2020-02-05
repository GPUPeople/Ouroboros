#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

template <typename DataType>
struct DeletionMarker
{
	static constexpr void* val{ nullptr };
};

template <>
struct DeletionMarker<index_t>
{
	static constexpr index_t val{ 0xFFFFFFFF };
};

template <>
struct DeletionMarker<unsigned long long>
{
	static constexpr unsigned long long val{ 0xFFFFFFFFFFFFFFFF };
};