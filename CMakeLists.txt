cmake_minimum_required(VERSION 3.18)

project(CudaMultiViewStereo)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "52;61;72;75;86")
endif()

add_subdirectory(modules)
add_subdirectory(samples)
