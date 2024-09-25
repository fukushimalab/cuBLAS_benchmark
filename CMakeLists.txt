cmake_minimum_required(VERSION 3.18)
project(main LANGUAGES CXX CUDA)

cmake_policy(SET CMP0146 NEW)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 )

add_executable(main main.cpp)

target_link_libraries(main
    PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    OpenMP::OpenMP_CXX
)

set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(main PRIVATE 
    -O3
    -fopenmp
    -mavx2
    -march=native
)