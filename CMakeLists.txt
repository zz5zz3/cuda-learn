cmake_minimum_required(VERSION 3.22)

project(cuda_learning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# ============================================================
# 公共函数：添加一个 CUDA 可执行程序
# ============================================================
function(add_cuda_demo target source)

    add_executable(${target}
        ${source}
    )

    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

endfunction()


# ============================================================
# Vector Add
# ============================================================
add_cuda_demo(vector_add
    operators/vector_add/main.cu
)


# ============================================================
# Reduction
# ============================================================
add_cuda_demo(reduction
    operators/reduction/main.cu
)

add_cuda_demo(WarpShuffleReduction
    operators/reduction/WarpShuffleReduction.cu
)

add_cuda_demo(WarpShuffleReduction_alltoone
    operators/reduction/WarpShuffleReduction_alltoone.cu
)


# ============================================================
# Matrix Transpose
# ============================================================
add_cuda_demo(MatrixTranspose
    operators/MatrixTranspose/MatrixTranspose.cu
)


# ============================================================
# GEMM
# ============================================================
add_cuda_demo(GEMM_native
    operators/GEMM/main.cu
)

add_cuda_demo(GEMM_memory
    operators/GEMM/Tiled_GEMM.cu
)

add_cuda_demo(GEMM_Register
    operators/GEMM/Register_GEMM.cu
)


# ============================================================
# Softmax
# ============================================================
add_cuda_demo(Softmax
    operators/softmax/main.cu
)

# ============================================================
# LayerNorm
# ============================================================
add_cuda_demo(LayerNorm
    operators/LayerNorm/main.cu
)