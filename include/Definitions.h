#pragma once

#include <typeinfo>
#include <memory>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

using memory_t = int8_t;
using index_t = uint32_t;

static constexpr int CACHELINE_SIZE{ 128 };
static constexpr int WARP_SIZE{ 32 };
static constexpr int UNLOCK{ 0 };
static constexpr int LOCK{ 1 };
static constexpr unsigned int BYTE_SIZE{ 8 };
static constexpr unsigned int CHUNK_IDENTIFIER{ std::numeric_limits<unsigned int>::max() };
static constexpr unsigned int QUEUECHUNK_IDENTIFIER{ std::numeric_limits<unsigned int>::max() - 1 };
static constexpr int FALSE {0};
static constexpr int TRUE {1};

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


static constexpr char break_line[] = {"##########################################################################################################\n"};
static constexpr char break_line_red[] = {"\033[0;31m##########################################################################################################\033[0m\n"};
static constexpr char break_line_red_s[] = {"\033[0;31m##########################################################################################################\n"};
static constexpr char break_line_red_e[] = {"##########################################################################################################\033[0m\n"};
static constexpr char break_line_green[] = {"\033[0;32m##########################################################################################################\033[0m\n"};
static constexpr char break_line_green_s[] = {"\033[0;32m##########################################################################################################\n"};
static constexpr char break_line_green_e[] = {"##########################################################################################################\033[0m\n"};
static constexpr char break_line_blue[] = {"\033[0;34m##########################################################################################################\033[0m\n"};
static constexpr char break_line_blue_s[] = {"\033[0;34m##########################################################################################################\n"};
static constexpr char break_line_blue_e[] = {"##########################################################################################################\033[0m\n"};
static constexpr char break_line_purple[] = {"\033[0;35m##########################################################################################################\033[0m\n"};
static constexpr char break_line_purple_s[] = {"\033[0;35m##########################################################################################################\n"};
static constexpr char break_line_purple_e[] = {"##########################################################################################################\033[0m\n"};
static constexpr char break_line_lblue[] = {"\033[1;34m##########################################################################################################\033[0m\n"};
static constexpr char break_line_cyan[] = {"\033[0;36m##########################################################################################################\033[0m\n"};
static constexpr char break_line_cyan_s[] = {"\033[0;36m##########################################################################################################\n"};
static constexpr char break_line_cyan_e[] = {"##########################################################################################################\033[0m\n"};