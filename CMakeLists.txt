cmake_minimum_required(VERSION 3.22)

project(cuda_learning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add
    operators/vector_add/main.cu
)

target_include_directories(vector_add PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(reduction
    operators/reduction/main.cu
)

target_include_directories(reduction PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(WarpShuffleReduction
    operators/reduction/WarpShuffleReduction.cu
)

target_include_directories(WarpShuffleReduction  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(WarpShuffleReduction_alltoone
    operators/reduction/WarpShuffleReduction_alltoone.cu
)

target_include_directories(WarpShuffleReduction_alltoone  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)