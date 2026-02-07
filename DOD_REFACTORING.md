# Data-Oriented Design (DOD) Refactoring

## 概述

本文档描述了StudyCorr库向数据导向设计（DOD，Data-Oriented Design）风格的重构。DOD是一种编程范式，通过优化数据布局和访问模式来实现高性能计算。

## DOD核心原则

### 1. 结构体数组（SoA）vs 数组结构体（AoS）

**传统的AoS布局（Array of Structures）：**
```cpp
struct CudaPOI2D {
    float x, y;                    // 8 bytes
    DeformationVector2D deformation;  // 48 bytes
    Result2D result;                  // 24 bytes
    StrainVector2D strain;            // 12 bytes
    float subset_radius_x, subset_radius_y;  // 8 bytes
    // Total: 100 bytes per POI
};
CudaPOI2D pois[1000];  // 100KB, 访问x时会加载整个结构体
```

**DOD的SoA布局（Structure of Arrays）：**
```cpp
struct POI2D_SoA {
    float* x;           // 所有x坐标连续存储
    float* y;           // 所有y坐标连续存储
    float* u;           // 所有u变形参数连续存储
    // ... 其他组件
    int count;
};
```

### 2. 性能优势

#### 缓存效率
- **AoS**: 访问1000个POI的x坐标 → 需要加载100KB数据，但只使用4KB → **96%浪费**
- **SoA**: 访问1000个POI的x坐标 → 只加载4KB数据 → **100%利用率**

#### GPU性能
- **合并访问**: 同一warp中的32个线程访问连续内存
- **更少的寄存器压力**: 每个线程只加载需要的数据
- **更好的占用率**: 更多活跃warp可以同时运行

#### SIMD效率
- CPU向量化指令可以同时处理多个连续数据
- GPU warp可以更高效地执行coalesced memory access

## 实现细节

### 新增文件

1. **sc_poi_dod.h** - DOD数据结构定义
   - `POI2D_SoA`: 2D POI的SoA布局
   - `POI3D_SoA`: 3D POI的SoA布局
   - 内存管理函数声明

2. **sc_poi_dod.cpp** - DOD数据结构实现
   - 主机和设备内存分配
   - AoS ↔ SoA 转换函数
   - 主机-设备数据传输

3. **sc_icgn_dod.h** - DOD版本ICGN接口
   - `ICGN2D1BatchGpuDOD`: 1阶2D ICGN
   - `ICGN2D2BatchGpuDOD`: 2阶2D ICGN
   - `ICGN3D1BatchGpuDOD`: 1阶3D ICGN

4. **sc_icgn_dod.cu** - DOD版本ICGN内核实现
   - `icgn2d_batch_kernel_dod`: 使用SoA的2D内核
   - `icgn3d1_batch_kernel_dod`: 使用SoA的3D内核

### 数据流程

```
1. 主机端（Host）:
   CudaPOI2D[] (AoS) → POI2D_SoA (SoA)
                         ↓
2. 传输到GPU:
   cudaMemcpyAsync (分别传输每个组件数组)
                         ↓
3. GPU计算:
   icgn2d_batch_kernel_dod (高效的SoA访问)
                         ↓
4. 传输回主机:
   cudaMemcpyAsync (分别传输结果数组)
                         ↓
5. 主机端（Host）:
   POI2D_SoA (SoA) → CudaPOI2D[] (AoS)
```

## 使用示例

### 基本使用

```cpp
#include "sc_poi_dod.h"
#include "sc_icgn_dod.h"

using namespace StudyCorr_GPU;

// 1. 准备输入数据（传统AoS格式）
std::vector<CudaPOI2D> pois_aos(1000);
// ... 初始化POI ...

// 2. 转换为SoA格式
POI2D_SoA host_soa, device_soa;
allocate_poi2d_soa_host(host_soa, 1000);
allocate_poi2d_soa_device(device_soa, 1000);
convert_aos_to_soa_2d(pois_aos.data(), host_soa, 1000);

// 3. 传输到GPU
copy_poi2d_soa_to_device(host_soa, device_soa);

// 4. 创建DOD计算对象并计算
ICGN2D2BatchGpuDOD icgn_dod;
ICGNParam param;
param.subsetRadius = 21;
param.convergenceThreshold = 0.001;
param.maxIterations = 10;

icgn_dod.prepare_cuda(ref_img, tar_img, height, width, param);
icgn_dod.compute_batch_cuda(device_soa);

// 5. 传输回主机并转换
copy_poi2d_soa_to_host(device_soa, host_soa);
convert_soa_to_aos_2d(host_soa, pois_aos.data(), 1000);

// 6. 清理
icgn_dod.release_cuda();
free_poi2d_soa(host_soa);
free_poi2d_soa(device_soa);
```

### 异步处理（更高性能）

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 准备多个批次
std::vector<POI2D_SoA> batches(4);
for (int i = 0; i < 4; i++) {
    allocate_poi2d_soa_device(batches[i], 256);
}

// 流水线处理
for (int i = 0; i < batches.size(); i++) {
    copy_poi2d_soa_to_device(host_batches[i], batches[i], stream);
    icgn_dod.compute_batch_cuda(batches[i], stream);
    copy_poi2d_soa_to_host(batches[i], result_batches[i], stream);
}

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

## 性能对比

### 理论分析

| 指标 | AoS | SoA | 改善 |
|------|-----|-----|------|
| 内存带宽利用率 | ~4% | ~100% | **25x** |
| 缓存命中率 | 低 | 高 | **5-10x** |
| GPU合并访问 | 无 | 是 | **3-8x** |
| 寄存器压力 | 高 | 低 | **1.5-2x** |

### 预期性能提升

对于典型的DIC计算场景（1000个POI，21像素子集）：

- **2D ICGN**: 预期加速 **3-5倍**
- **3D ICGN**: 预期加速 **4-8倍**（3D数据更大，改善更明显）
- **内存传输**: 预期加速 **2-3倍**（减少无效数据传输）

## 向后兼容

### 保持现有接口

原有的AoS接口和类保持不变：
- `CudaPOI2D`, `CudaPOI3D` 结构体
- `ICGN2D1BatchGpu`, `ICGN2D2BatchGpu`, `ICGN3D1BatchGpu` 类
- 所有现有代码无需修改

### 渐进式迁移

用户可以选择：
1. **继续使用AoS**: 保持现有代码不变
2. **迁移到SoA**: 获得更高性能
3. **混合使用**: 在性能关键路径使用SoA

## 最佳实践

### 何时使用DOD/SoA

✅ **适合使用SoA的场景**:
- 大批量POI处理（>100个）
- 计算密集型操作（ICGN迭代）
- 需要最高性能的场景
- GPU计算为主

❌ **不适合使用SoA的场景**:
- 小批量处理（<50个POI）
- 频繁的AoS↔SoA转换
- 以CPU处理为主的场景

### 性能调优建议

1. **批量大小**: 256-1024个POI为最佳批量
2. **使用CUDA流**: 重叠数据传输和计算
3. **预分配内存**: 避免重复分配/释放
4. **零拷贝优化**: 对于pinned memory使用异步传输

## 未来工作

### 待完成的优化

- [ ] 完善3D ICGN内核实现
- [ ] 添加共享内存优化
- [ ] 实现warp-level primitives
- [ ] 添加性能基准测试
- [ ] 优化数据传输策略

### 扩展计划

- [ ] 支持更多算法的DOD版本
  - SIFT特征提取
  - 梯度计算
  - 应变计算
- [ ] CPU端的SIMD优化
- [ ] 多GPU支持

## 参考资料

1. [Data-Oriented Design Book](https://www.dataorienteddesign.com/)
2. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [Structure of Arrays (SoA) in CUDA](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-data-transfers/)

## 贡献指南

如果你想为DOD重构贡献代码：

1. 保持SoA接口的一致性
2. 添加单元测试验证正确性
3. 进行性能基准测试
4. 更新相关文档

## 许可证

本项目遵循原StudyCorr项目的许可证。
