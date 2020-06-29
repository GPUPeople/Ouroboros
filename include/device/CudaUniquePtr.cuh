#pragma once
#include "Definitions.h"

// #######################################################################################
// Unique pointer for device memory
template <typename DataType>
struct CudaUniquePtr
{
	// #######################################################################################
	// Data members
	CudaUniquePtr() : size{ 0 }, data{ nullptr }{}

	CudaUniquePtr(size_t new_size) : size{ new_size }
	{
		HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(DataType)));
	}

	~CudaUniquePtr()
	{
		if (data)
			HANDLE_ERROR(cudaFree(data));
	}

	// Disallow copy of any kind, only allow move
	CudaUniquePtr(const CudaUniquePtr&) = delete;
	CudaUniquePtr& operator=(const CudaUniquePtr&) = delete;
	CudaUniquePtr(CudaUniquePtr&& other) noexcept : data{ std::exchange(other.data, nullptr) }, size{ std::exchange(other.size, 0) }{}
	CudaUniquePtr& operator=(CudaUniquePtr&& other) noexcept
	{
		std::swap(data, other.data);
		std::swap(size, other.size);
		return *this;
	}

	// #######################################################################################
	// 
	void allocate(size_t new_size)
	{
		if (data && size != new_size)
		{
			HANDLE_ERROR(cudaFree(data));
			data = nullptr;
		}

		size = new_size;
		if (data == nullptr)
			HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(DataType)));
	}

	// #######################################################################################
	// 
	void resize(size_t new_size)
	{
		void* tmp{nullptr};
		HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&tmp), new_size * sizeof(DataType)));
		HANDLE_ERROR(cudaMemcpy(tmp, data, sizeof(DataType) * std::min(size, new_size), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaFree(data));
		size = new_size;
		data = reinterpret_cast<DataType*>(tmp);
	}

	// #######################################################################################
	// 
	void copyToDevice(DataType* host_data, size_t copy_size, unsigned int offset = 0)
	{
		HANDLE_ERROR(cudaMemcpy(data + offset, host_data, sizeof(DataType) * copy_size, cudaMemcpyHostToDevice));
	}

	// #######################################################################################
	// 
	void copyFromDevice(DataType* host_data, size_t copy_size, unsigned int offset = 0)
	{
		HANDLE_ERROR(cudaMemcpy(host_data, data + offset, sizeof(DataType) * copy_size, cudaMemcpyDeviceToHost));
	}

	// #######################################################################################
	// 
	void memSet(DataType value, size_t memset_size)
	{
		HANDLE_ERROR(cudaMemset(data, value, sizeof(DataType) * memset_size));
	}

	// #######################################################################################
	// 
	explicit operator DataType*() { return data; }

	// #######################################################################################
	// 
	DataType* get() { return data; }

	// #######################################################################################
	// 
	void release()
	{
		if (data)
		{
			HANDLE_ERROR(cudaFree(data));
			data = nullptr;
			size = 0;
		}

	}

	// #######################################################################################
	// Data members
	DataType* data{ nullptr };
	size_t size{ 0 };
};