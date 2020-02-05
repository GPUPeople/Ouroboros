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

add_executable(main
          src/main.cu
)