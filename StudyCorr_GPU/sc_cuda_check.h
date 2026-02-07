/*
 * CUDA error checking utilities for DOD implementation
 */

#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace StudyCorr_GPU {

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Alternative: throw exception instead of exit
#define CUDA_CHECK_THROW(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            char error_msg[256]; \
            snprintf(error_msg, sizeof(error_msg), "CUDA error at %s:%d: %s", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

} // namespace StudyCorr_GPU
