#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <cassert>

// CUDA allocator for managed memory
template<typename T>
struct CudaAllocator
{
    using value_type = T;

    T* allocate(std::size_t size)
    {
        T* ptr = nullptr;
        cudaMallocManaged(&ptr, size * sizeof(T));
        return ptr;
    }

    void deallocate(T* ptr, std::size_t = 0)
    {
        cudaFree(ptr);
    }

    template <class ...Args>
    void construct(T* ptr, Args&&... args)
    {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void*)ptr) T(std::forward<Args>(args)...);
    }
};

//1D CUDA array
template<typename T>
struct Array1D
{
    T* data = nullptr;
    size_t size = 0;

    Array1D() = default;
    Array1D(size_t size_) : size(size_)
    {
        data = CudaAllocator<T>().allocate(size_);
        assert(data);
    }
    Array1D(size_t size_, const T& value) : size(size_)
    {
    data = CudaAllocator<T>().allocate(size_);
    assert(data);
    for (size_t i = 0; i < size_; ++i) data[i] = value;
    }
    ~Array1D()
    {
        if (data) CudaAllocator<T>().deallocate(data);
    }
    __host__ __device__
    T& operator[](size_t i) { return data[i]; }
    __host__ __device__
    const T& operator[](size_t i) const { return data[i]; }
    T* get() { return data; }
    const T* get() const { return data; }
    size_t dim() const { return size; }
};


// 2D CUDA array
template<typename T>
struct Array2D
{
    T* data = nullptr;
    size_t rows = 0, cols = 0;

    Array2D() = default;
    Array2D(size_t rows_, size_t cols_) : rows(rows_), cols(cols_)
    {
        data = CudaAllocator<T>().allocate(rows_ * cols_);
        assert(data);
    }
    Array2D(size_t rows_, size_t cols_, const T& value) : rows(rows_), cols(cols_)
    {
        data = CudaAllocator<T>().allocate(rows_ * cols_);
        assert(data);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                data[i * cols + j] = value;
    }

    ~Array2D()
    {
        if (data) CudaAllocator<T>().deallocate(data);
    }

    __host__ __device__
    T& operator()(size_t i, size_t j) { return data[i * cols + j]; }

    __host__ __device__
    const T& operator()(size_t i, size_t j) const { return data[i * cols + j]; }

    T* get() { return data; }
    const T* get() const { return data; }
    size_t dim1() const { return rows; }
    size_t dim2() const { return cols; }
};

// 3D CUDA array
template<typename T>
struct Array3D //[i][j][k],最外层 i，中层 j，最内层 k,i 个块（每块 d2*d3），每块中 j 行（每行 d3），每行 k 个元素
{
    T* data = nullptr;
    size_t d1 = 0, d2 = 0, d3 = 0;

    Array3D() = default;
    Array3D(size_t d1_, size_t d2_, size_t d3_) : d1(d1_), d2(d2_), d3(d3_)
    {
        data = CudaAllocator<T>().allocate(d1_ * d2_ * d3_);
        assert(data);
    }
    Array3D(size_t d1_, size_t d2_, size_t d3_, const T& value) : d1(d1_), d2(d2_), d3(d3_)
    {
        data = CudaAllocator<T>().allocate(d1_ * d2_ * d3_);
        assert(data);
        for (size_t i = 0; i < d1_; ++i)
            for (size_t j = 0; j < d2_; ++j)
                for (size_t k = 0; k < d3_; ++k)
                    data[(i * d2 + j) * d3 + k] = value;
    }

    ~Array3D()
    {
        if (data) CudaAllocator<T>().deallocate(data);
    }

    __host__ __device__
    T& operator()(size_t i, size_t j, size_t k) { return data[(i * d2 + j) * d3 + k]; }

    __host__ __device__
    const T& operator()(size_t i, size_t j, size_t k) const { return data[(i * d2 + j) * d3 + k]; }

    T* get() { return data; }
    const T* get() const { return data; }
    size_t dim1() const { return d1; }
    size_t dim2() const { return d2; }
    size_t dim3() const { return d3; }
};

// 4D CUDA array
template<typename T>
struct Array4D
{
    T* data = nullptr;
    size_t d1 = 0, d2 = 0, d3 = 0, d4 = 0;

    Array4D() = default;
    Array4D(size_t d1_, size_t d2_, size_t d3_, size_t d4_) : d1(d1_), d2(d2_), d3(d3_), d4(d4_)
    {
        data = CudaAllocator<T>().allocate(d1_ * d2_ * d3_ * d4_);
        assert(data);
    }

    Array4D(size_t d1_, size_t d2_, size_t d3_, size_t d4_, const T& value) : d1(d1_), d2(d2_), d3(d3_), d4(d4_)
    {
        data = CudaAllocator<T>().allocate(d1_ * d2_ * d3_ * d4_);
        assert(data);
        for (size_t i = 0; i < d1_; ++i)
            for (size_t j = 0; j < d2_; ++j)
                for (size_t k = 0; k < d3_; ++k)
                    for (size_t l = 0; l < d4_; ++l)
                        data[((i * d2 + j) * d3 + k) * d4 + l] = value;
    }

    ~Array4D()
    {
        if (data) CudaAllocator<T>().deallocate(data);
    }

    __host__ __device__
    T& operator()(size_t i, size_t j, size_t k, size_t l) { return data[((i * d2 + j) * d3 + k) * d4 + l]; }

    __host__ __device__
    const T& operator()(size_t i, size_t j, size_t k, size_t l) const { return data[((i * d2 + j) * d3 + k) * d4 + l]; }

    T* get() { return data; }
    const T* get() const { return data; }
    size_t dim1() const { return d1; }
    size_t dim2() const { return d2; }
    size_t dim3() const { return d3; }
    size_t dim4() const { return d4; }
};



// #include "cuda_array.h"

// __global__ void kernel(CudaArray2D<float> arr, int rows, int cols)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < rows && j < cols)
//         arr(i, j) = arr(i, j) * 2.0f;
// }

// int main()
// {
//     CudaArray2D<float> arr(10, 20);
//     arr(0, 0) = 1.0f;
//
//     dim3 block(8, 8);
//     dim3 grid((10+block.x-1)/block.x, (20+block.y-1)/block.y);
//     kernel<<<grid, block>>>(arr, 10, 20);
//     cudaDeviceSynchronize();//一定要有这个，要不然会出现未定义行为，cpu端可能会提前访问未更新的值
// }