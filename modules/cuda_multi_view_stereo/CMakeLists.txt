cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME cuda_multi_view_stereo)
set(INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INTERNAL_INCLUDE_DIR ${INTERFACE_INCLUDE_DIR})

# create project
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# dependent packages
find_package(OpenCV REQUIRED core cudev)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# target configuration
file(GLOB SRCS src/*.cpp src/*.cu src/*.h* ${INTERNAL_INCLUDE_DIR}/*.h*)
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${INTERNAL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX CUDA::cudart)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${INTERFACE_INCLUDE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-lineinfo -Xcompiler="\"/wd4819 /wd4244\"">
)
