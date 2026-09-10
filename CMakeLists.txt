cmake_minimum_required(VERSION 3.18)
project(ZernikePhaseUnwrap LANGUAGES CXX CUDA)

# 1. 强制启用 Release 模式 (解决 CMAKE_BUILD_TYPE 未指定问题)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. 注入顶级编译器优化标志
# --use_fast_math: GPU 浮点/三角函数硬件加速
# -Xcompiler -O3,-march=native: 传给 CPU 编译器 (GCC) 进行本机 CPU 指令集最高级优化
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -O3,-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_ARCHITECTURES 89)

# 查找 CUDA 库
find_package(CUDAToolkit REQUIRED)

# 尝试查找 OpenMP (如果 CPU 代码有 #pragma omp)
find_package(OpenMP)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    kernel_testbench.cu
)

add_executable(app ${SOURCES})

target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# 链接 OpenMP (如果找到的话)
if(OpenMP_CXX_FOUND)
    target_link_libraries(app PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(app PRIVATE
    CUDA::toolkit
    CUDA::cublas
    CUDA::cusolver
    CUDA::cufft
)