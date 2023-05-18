cmake_minimum_required(VERSION 3.20)

project(master-simarpn++)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)


# cuda-tensorrt-libtorch-opencv dictionary
set(TRT_DIR D:/TensorRT-8.5.1.7)
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
set(OpenCV_DIR  D:/opencv/build)
set(Libtorch_DIR D:/libtorch)


# cuda
include_directories(${CUDA_DIR}/v11.4/include)
link_directories(${CUDA_DIR}/v11.4/lib/x64)
find_package(CUDA REQUIRED)

# tensorrt
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)


# opencv
include_directories(${OpenCV_DIR}/x64/vc14/lib)
include_directories(${OpenCV_DIR}/include)
include_directories(${OpenCV_DIR}/include/opencv2)
include_directories(${OpenCV_DIR}/include/opencv2/core)
find_package(OpenCV REQUIRED)

# Eigen
include_directories(D:/eigen-3.3.9)

# libtorch
include_directories(${Libtorch_DIR}/include)
include_directories(${Libtorch_DIR}/include/torch/csrc/api/include)
link_directories(${Libtorch_DIR}/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--extended-lambda)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS}; -std=c++17; -g; -G; -gencode; arch=compute_75; code=sm_75)
set(CMAKE_CUDA_COMPILER  ${CUDA_DIR}/v11.4/bin)
set(SRCS ${PROJECT_SOURCE_DIR}/common.h
			${PROJECT_SOURCE_DIR}/cuda_utils.h
			${PROJECT_SOURCE_DIR}/simaRPN++.h
			${PROJECT_SOURCE_DIR}/main.cpp
			${PROJECT_SOURCE_DIR}/simaRPN++.cpp
			${PROJECT_SOURCE_DIR}/common.cpp)
			
set(DEPS nvonnxparser
         libprotobufd
         nvinfer
         nvinfer_plugin
         nvparsers
         cudart_static
         cudart
         c10_cuda
         c10
         mkldnn
         torch
         torch_cuda
         torch_cpu
		 ${OpenCV_LIBS})

add_executable(master-simarpn++         ${SRCS})
target_link_libraries(master-simarpn++  ${DEPS})