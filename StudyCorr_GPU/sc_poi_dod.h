/*
 * Data-Oriented Design (DOD) version of POI structures
 * Using Structure of Arrays (SoA) layout for better cache performance
 * and GPU memory access patterns
 */

#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "sc_point.h"

namespace StudyCorr_GPU
{
    // Structure of Arrays for 2D POI data
    // Separates data by component for better memory access patterns
    struct POI2D_SoA
    {
        // Position data
        float* x;
        float* y;
        
        // Deformation parameters (12 components for 2nd order)
        float* u;
        float* ux;
        float* uy;
        float* uxx;
        float* uxy;
        float* uyy;
        float* v;
        float* vx;
        float* vy;
        float* vxx;
        float* vxy;
        float* vyy;
        
        // Result data (6 components)
        float* u0;      // Initial u estimate
        float* v0;      // Initial v estimate
        float* zncc;    // Zero-mean Normalized Cross-Correlation
        float* iteration;
        float* convergence;
        float* feature;
        
        // Strain data (3 components for 2D)
        float* exx;
        float* eyy;
        float* exy;
        
        // Subset radius
        float* subset_radius_x;
        float* subset_radius_y;
        
        // Number of POIs
        int count;
        
        // Memory management flag
        bool is_device_memory;
        
        // Constructor
        POI2D_SoA() : count(0), is_device_memory(false),
            x(nullptr), y(nullptr),
            u(nullptr), ux(nullptr), uy(nullptr), uxx(nullptr), uxy(nullptr), uyy(nullptr),
            v(nullptr), vx(nullptr), vy(nullptr), vxx(nullptr), vxy(nullptr), vyy(nullptr),
            u0(nullptr), v0(nullptr), zncc(nullptr), iteration(nullptr), convergence(nullptr), feature(nullptr),
            exx(nullptr), eyy(nullptr), exy(nullptr),
            subset_radius_x(nullptr), subset_radius_y(nullptr) {}
    };
    
    // Structure of Arrays for 3D POI data
    struct POI3D_SoA
    {
        // Position data
        float* x;
        float* y;
        float* z;
        
        // Deformation parameters (12 components for 1st order 3D)
        float* u;
        float* ux;
        float* uy;
        float* uz;
        float* v;
        float* vx;
        float* vy;
        float* vz;
        float* w;
        float* wx;
        float* wy;
        float* wz;
        
        // Result data (7 components)
        float* u0;
        float* v0;
        float* w0;
        float* zncc;
        float* iteration;
        float* convergence;
        float* feature;
        
        // Strain data (6 components for 3D)
        float* exx;
        float* eyy;
        float* ezz;
        float* exy;
        float* eyz;
        float* ezx;
        
        // Subset radius
        float* subset_radius_x;
        float* subset_radius_y;
        float* subset_radius_z;
        
        // Number of POIs
        int count;
        
        // Memory management flag
        bool is_device_memory;
        
        // Constructor
        POI3D_SoA() : count(0), is_device_memory(false),
            x(nullptr), y(nullptr), z(nullptr),
            u(nullptr), ux(nullptr), uy(nullptr), uz(nullptr),
            v(nullptr), vx(nullptr), vy(nullptr), vz(nullptr),
            w(nullptr), wx(nullptr), wy(nullptr), wz(nullptr),
            u0(nullptr), v0(nullptr), w0(nullptr), zncc(nullptr), iteration(nullptr), convergence(nullptr), feature(nullptr),
            exx(nullptr), eyy(nullptr), ezz(nullptr), exy(nullptr), eyz(nullptr), ezx(nullptr),
            subset_radius_x(nullptr), subset_radius_y(nullptr), subset_radius_z(nullptr) {}
    };
    
    // Helper functions for memory management
    
    // Allocate POI2D_SoA on host
    void allocate_poi2d_soa_host(POI2D_SoA& soa, int count);
    
    // Allocate POI2D_SoA on device
    void allocate_poi2d_soa_device(POI2D_SoA& soa, int count);
    
    // Free POI2D_SoA memory
    void free_poi2d_soa(POI2D_SoA& soa);
    
    // Allocate POI3D_SoA on host
    void allocate_poi3d_soa_host(POI3D_SoA& soa, int count);
    
    // Allocate POI3D_SoA on device
    void allocate_poi3d_soa_device(POI3D_SoA& soa, int count);
    
    // Free POI3D_SoA memory
    void free_poi3d_soa(POI3D_SoA& soa);
    
    // Conversion functions between AoS and SoA
    
    // Convert from Array of Structures to Structure of Arrays (2D)
    void convert_aos_to_soa_2d(const CudaPOI2D* aos, POI2D_SoA& soa, int count);
    
    // Convert from Structure of Arrays to Array of Structures (2D)
    void convert_soa_to_aos_2d(const POI2D_SoA& soa, CudaPOI2D* aos, int count);
    
    // Convert from Array of Structures to Structure of Arrays (3D)
    void convert_aos_to_soa_3d(const CudaPOI3D* aos, POI3D_SoA& soa, int count);
    
    // Convert from Structure of Arrays to Array of Structures (3D)
    void convert_soa_to_aos_3d(const POI3D_SoA& soa, CudaPOI3D* aos, int count);
    
    // Copy POI2D_SoA from host to device
    void copy_poi2d_soa_to_device(const POI2D_SoA& host_soa, POI2D_SoA& device_soa, cudaStream_t stream = 0);
    
    // Copy POI2D_SoA from device to host
    void copy_poi2d_soa_to_host(const POI2D_SoA& device_soa, POI2D_SoA& host_soa, cudaStream_t stream = 0);
    
    // Copy POI3D_SoA from host to device
    void copy_poi3d_soa_to_device(const POI3D_SoA& host_soa, POI3D_SoA& device_soa, cudaStream_t stream = 0);
    
    // Copy POI3D_SoA from device to host
    void copy_poi3d_soa_to_host(const POI3D_SoA& device_soa, POI3D_SoA& host_soa, cudaStream_t stream = 0);
    
} // namespace StudyCorr_GPU
