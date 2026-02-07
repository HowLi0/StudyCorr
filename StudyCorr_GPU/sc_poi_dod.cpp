/*
 * Implementation of DOD POI memory management and conversion functions
 */

#include "sc_poi_dod.h"
#include "sc_poi.h"
#include "sc_cuda_check.h"
#include <cuda_runtime.h>
#include <cstring>

namespace StudyCorr_GPU
{
    // Allocate POI2D_SoA on host
    void allocate_poi2d_soa_host(POI2D_SoA& soa, int count)
    {
        soa.count = count;
        soa.is_device_memory = false;
        
        // Allocate position data
        soa.x = new float[count];
        soa.y = new float[count];
        
        // Allocate deformation parameters
        soa.u = new float[count];
        soa.ux = new float[count];
        soa.uy = new float[count];
        soa.uxx = new float[count];
        soa.uxy = new float[count];
        soa.uyy = new float[count];
        soa.v = new float[count];
        soa.vx = new float[count];
        soa.vy = new float[count];
        soa.vxx = new float[count];
        soa.vxy = new float[count];
        soa.vyy = new float[count];
        
        // Allocate result data
        soa.u0 = new float[count];
        soa.v0 = new float[count];
        soa.zncc = new float[count];
        soa.iteration = new float[count];
        soa.convergence = new float[count];
        soa.feature = new float[count];
        
        // Allocate strain data
        soa.exx = new float[count];
        soa.eyy = new float[count];
        soa.exy = new float[count];
        
        // Allocate subset radius
        soa.subset_radius_x = new float[count];
        soa.subset_radius_y = new float[count];
    }
    
    // Allocate POI2D_SoA on device
    void allocate_poi2d_soa_device(POI2D_SoA& soa, int count)
    {
        soa.count = count;
        soa.is_device_memory = true;
        
        size_t size = count * sizeof(float);
        
        // Allocate position data
        CUDA_CHECK(cudaMalloc(&soa.x, size));
        CUDA_CHECK(cudaMalloc(&soa.y, size));
        
        // Allocate deformation parameters
        CUDA_CHECK(cudaMalloc(&soa.u, size));
        CUDA_CHECK(cudaMalloc(&soa.ux, size));
        CUDA_CHECK(cudaMalloc(&soa.uy, size));
        CUDA_CHECK(cudaMalloc(&soa.uxx, size));
        CUDA_CHECK(cudaMalloc(&soa.uxy, size));
        CUDA_CHECK(cudaMalloc(&soa.uyy, size));
        CUDA_CHECK(cudaMalloc(&soa.v, size));
        CUDA_CHECK(cudaMalloc(&soa.vx, size));
        CUDA_CHECK(cudaMalloc(&soa.vy, size));
        CUDA_CHECK(cudaMalloc(&soa.vxx, size));
        CUDA_CHECK(cudaMalloc(&soa.vxy, size));
        CUDA_CHECK(cudaMalloc(&soa.vyy, size));
        
        // Allocate result data
        CUDA_CHECK(cudaMalloc(&soa.u0, size));
        CUDA_CHECK(cudaMalloc(&soa.v0, size));
        CUDA_CHECK(cudaMalloc(&soa.zncc, size));
        CUDA_CHECK(cudaMalloc(&soa.iteration, size));
        CUDA_CHECK(cudaMalloc(&soa.convergence, size));
        CUDA_CHECK(cudaMalloc(&soa.feature, size));
        
        // Allocate strain data
        CUDA_CHECK(cudaMalloc(&soa.exx, size));
        CUDA_CHECK(cudaMalloc(&soa.eyy, size));
        CUDA_CHECK(cudaMalloc(&soa.exy, size));
        
        // Allocate subset radius
        CUDA_CHECK(cudaMalloc(&soa.subset_radius_x, size));
        CUDA_CHECK(cudaMalloc(&soa.subset_radius_y, size));
    }
    
    // Free POI2D_SoA memory
    void free_poi2d_soa(POI2D_SoA& soa)
    {
        if (soa.count == 0) return;
        
        if (soa.is_device_memory)
        {
            // Free device memory
            cudaFree(soa.x);
            cudaFree(soa.y);
            cudaFree(soa.u);
            cudaFree(soa.ux);
            cudaFree(soa.uy);
            cudaFree(soa.uxx);
            cudaFree(soa.uxy);
            cudaFree(soa.uyy);
            cudaFree(soa.v);
            cudaFree(soa.vx);
            cudaFree(soa.vy);
            cudaFree(soa.vxx);
            cudaFree(soa.vxy);
            cudaFree(soa.vyy);
            cudaFree(soa.u0);
            cudaFree(soa.v0);
            cudaFree(soa.zncc);
            cudaFree(soa.iteration);
            cudaFree(soa.convergence);
            cudaFree(soa.feature);
            cudaFree(soa.exx);
            cudaFree(soa.eyy);
            cudaFree(soa.exy);
            cudaFree(soa.subset_radius_x);
            cudaFree(soa.subset_radius_y);
        }
        else
        {
            // Free host memory
            delete[] soa.x;
            delete[] soa.y;
            delete[] soa.u;
            delete[] soa.ux;
            delete[] soa.uy;
            delete[] soa.uxx;
            delete[] soa.uxy;
            delete[] soa.uyy;
            delete[] soa.v;
            delete[] soa.vx;
            delete[] soa.vy;
            delete[] soa.vxx;
            delete[] soa.vxy;
            delete[] soa.vyy;
            delete[] soa.u0;
            delete[] soa.v0;
            delete[] soa.zncc;
            delete[] soa.iteration;
            delete[] soa.convergence;
            delete[] soa.feature;
            delete[] soa.exx;
            delete[] soa.eyy;
            delete[] soa.exy;
            delete[] soa.subset_radius_x;
            delete[] soa.subset_radius_y;
        }
        
        soa.count = 0;
    }
    
    // Allocate POI3D_SoA on host
    void allocate_poi3d_soa_host(POI3D_SoA& soa, int count)
    {
        soa.count = count;
        soa.is_device_memory = false;
        
        // Allocate position data
        soa.x = new float[count];
        soa.y = new float[count];
        soa.z = new float[count];
        
        // Allocate deformation parameters
        soa.u = new float[count];
        soa.ux = new float[count];
        soa.uy = new float[count];
        soa.uz = new float[count];
        soa.v = new float[count];
        soa.vx = new float[count];
        soa.vy = new float[count];
        soa.vz = new float[count];
        soa.w = new float[count];
        soa.wx = new float[count];
        soa.wy = new float[count];
        soa.wz = new float[count];
        
        // Allocate result data
        soa.u0 = new float[count];
        soa.v0 = new float[count];
        soa.w0 = new float[count];
        soa.zncc = new float[count];
        soa.iteration = new float[count];
        soa.convergence = new float[count];
        soa.feature = new float[count];
        
        // Allocate strain data
        soa.exx = new float[count];
        soa.eyy = new float[count];
        soa.ezz = new float[count];
        soa.exy = new float[count];
        soa.eyz = new float[count];
        soa.ezx = new float[count];
        
        // Allocate subset radius
        soa.subset_radius_x = new float[count];
        soa.subset_radius_y = new float[count];
        soa.subset_radius_z = new float[count];
    }
    
    // Allocate POI3D_SoA on device
    void allocate_poi3d_soa_device(POI3D_SoA& soa, int count)
    {
        soa.count = count;
        soa.is_device_memory = true;
        
        size_t size = count * sizeof(float);
        
        // Allocate position data
        CUDA_CHECK(cudaMalloc(&soa.x, size));
        CUDA_CHECK(cudaMalloc(&soa.y, size));
        CUDA_CHECK(cudaMalloc(&soa.z, size));
        
        // Allocate deformation parameters
        CUDA_CHECK(cudaMalloc(&soa.u, size));
        CUDA_CHECK(cudaMalloc(&soa.ux, size));
        CUDA_CHECK(cudaMalloc(&soa.uy, size));
        CUDA_CHECK(cudaMalloc(&soa.uz, size));
        CUDA_CHECK(cudaMalloc(&soa.v, size));
        CUDA_CHECK(cudaMalloc(&soa.vx, size));
        CUDA_CHECK(cudaMalloc(&soa.vy, size));
        CUDA_CHECK(cudaMalloc(&soa.vz, size));
        CUDA_CHECK(cudaMalloc(&soa.w, size));
        CUDA_CHECK(cudaMalloc(&soa.wx, size));
        CUDA_CHECK(cudaMalloc(&soa.wy, size));
        CUDA_CHECK(cudaMalloc(&soa.wz, size));
        
        // Allocate result data
        CUDA_CHECK(cudaMalloc(&soa.u0, size));
        CUDA_CHECK(cudaMalloc(&soa.v0, size));
        CUDA_CHECK(cudaMalloc(&soa.w0, size));
        CUDA_CHECK(cudaMalloc(&soa.zncc, size));
        CUDA_CHECK(cudaMalloc(&soa.iteration, size));
        CUDA_CHECK(cudaMalloc(&soa.convergence, size));
        CUDA_CHECK(cudaMalloc(&soa.feature, size));
        
        // Allocate strain data
        CUDA_CHECK(cudaMalloc(&soa.exx, size));
        CUDA_CHECK(cudaMalloc(&soa.eyy, size));
        CUDA_CHECK(cudaMalloc(&soa.ezz, size));
        CUDA_CHECK(cudaMalloc(&soa.exy, size));
        CUDA_CHECK(cudaMalloc(&soa.eyz, size));
        CUDA_CHECK(cudaMalloc(&soa.ezx, size));
        
        // Allocate subset radius
        CUDA_CHECK(cudaMalloc(&soa.subset_radius_x, size));
        CUDA_CHECK(cudaMalloc(&soa.subset_radius_y, size));
        CUDA_CHECK(cudaMalloc(&soa.subset_radius_z, size));
    }
    
    // Free POI3D_SoA memory
    void free_poi3d_soa(POI3D_SoA& soa)
    {
        if (soa.count == 0) return;
        
        if (soa.is_device_memory)
        {
            // Free device memory
            cudaFree(soa.x);
            cudaFree(soa.y);
            cudaFree(soa.z);
            cudaFree(soa.u);
            cudaFree(soa.ux);
            cudaFree(soa.uy);
            cudaFree(soa.uz);
            cudaFree(soa.v);
            cudaFree(soa.vx);
            cudaFree(soa.vy);
            cudaFree(soa.vz);
            cudaFree(soa.w);
            cudaFree(soa.wx);
            cudaFree(soa.wy);
            cudaFree(soa.wz);
            cudaFree(soa.u0);
            cudaFree(soa.v0);
            cudaFree(soa.w0);
            cudaFree(soa.zncc);
            cudaFree(soa.iteration);
            cudaFree(soa.convergence);
            cudaFree(soa.feature);
            cudaFree(soa.exx);
            cudaFree(soa.eyy);
            cudaFree(soa.ezz);
            cudaFree(soa.exy);
            cudaFree(soa.eyz);
            cudaFree(soa.ezx);
            cudaFree(soa.subset_radius_x);
            cudaFree(soa.subset_radius_y);
            cudaFree(soa.subset_radius_z);
        }
        else
        {
            // Free host memory
            delete[] soa.x;
            delete[] soa.y;
            delete[] soa.z;
            delete[] soa.u;
            delete[] soa.ux;
            delete[] soa.uy;
            delete[] soa.uz;
            delete[] soa.v;
            delete[] soa.vx;
            delete[] soa.vy;
            delete[] soa.vz;
            delete[] soa.w;
            delete[] soa.wx;
            delete[] soa.wy;
            delete[] soa.wz;
            delete[] soa.u0;
            delete[] soa.v0;
            delete[] soa.w0;
            delete[] soa.zncc;
            delete[] soa.iteration;
            delete[] soa.convergence;
            delete[] soa.feature;
            delete[] soa.exx;
            delete[] soa.eyy;
            delete[] soa.ezz;
            delete[] soa.exy;
            delete[] soa.eyz;
            delete[] soa.ezx;
            delete[] soa.subset_radius_x;
            delete[] soa.subset_radius_y;
            delete[] soa.subset_radius_z;
        }
        
        soa.count = 0;
    }
    
    // Convert from Array of Structures to Structure of Arrays (2D)
    void convert_aos_to_soa_2d(const CudaPOI2D* aos, POI2D_SoA& soa, int count)
    {
        for (int i = 0; i < count; i++)
        {
            // Position
            soa.x[i] = aos[i].x;
            soa.y[i] = aos[i].y;
            
            // Deformation
            soa.u[i] = aos[i].deformation.u;
            soa.ux[i] = aos[i].deformation.ux;
            soa.uy[i] = aos[i].deformation.uy;
            soa.uxx[i] = aos[i].deformation.uxx;
            soa.uxy[i] = aos[i].deformation.uxy;
            soa.uyy[i] = aos[i].deformation.uyy;
            soa.v[i] = aos[i].deformation.v;
            soa.vx[i] = aos[i].deformation.vx;
            soa.vy[i] = aos[i].deformation.vy;
            soa.vxx[i] = aos[i].deformation.vxx;
            soa.vxy[i] = aos[i].deformation.vxy;
            soa.vyy[i] = aos[i].deformation.vyy;
            
            // Result
            soa.u0[i] = aos[i].result.u0;
            soa.v0[i] = aos[i].result.v0;
            soa.zncc[i] = aos[i].result.zncc;
            soa.iteration[i] = aos[i].result.iteration;
            soa.convergence[i] = aos[i].result.convergence;
            soa.feature[i] = aos[i].result.feature;
            
            // Strain
            soa.exx[i] = aos[i].strain.exx;
            soa.eyy[i] = aos[i].strain.eyy;
            soa.exy[i] = aos[i].strain.exy;
            
            // Subset radius
            soa.subset_radius_x[i] = aos[i].subset_radius_x;
            soa.subset_radius_y[i] = aos[i].subset_radius_y;
        }
    }
    
    // Convert from Structure of Arrays to Array of Structures (2D)
    void convert_soa_to_aos_2d(const POI2D_SoA& soa, CudaPOI2D* aos, int count)
    {
        for (int i = 0; i < count; i++)
        {
            // Position
            aos[i].x = soa.x[i];
            aos[i].y = soa.y[i];
            
            // Deformation
            aos[i].deformation.u = soa.u[i];
            aos[i].deformation.ux = soa.ux[i];
            aos[i].deformation.uy = soa.uy[i];
            aos[i].deformation.uxx = soa.uxx[i];
            aos[i].deformation.uxy = soa.uxy[i];
            aos[i].deformation.uyy = soa.uyy[i];
            aos[i].deformation.v = soa.v[i];
            aos[i].deformation.vx = soa.vx[i];
            aos[i].deformation.vy = soa.vy[i];
            aos[i].deformation.vxx = soa.vxx[i];
            aos[i].deformation.vxy = soa.vxy[i];
            aos[i].deformation.vyy = soa.vyy[i];
            
            // Result
            aos[i].result.u0 = soa.u0[i];
            aos[i].result.v0 = soa.v0[i];
            aos[i].result.zncc = soa.zncc[i];
            aos[i].result.iteration = soa.iteration[i];
            aos[i].result.convergence = soa.convergence[i];
            aos[i].result.feature = soa.feature[i];
            
            // Strain
            aos[i].strain.exx = soa.exx[i];
            aos[i].strain.eyy = soa.eyy[i];
            aos[i].strain.exy = soa.exy[i];
            
            // Subset radius
            aos[i].subset_radius_x = soa.subset_radius_x[i];
            aos[i].subset_radius_y = soa.subset_radius_y[i];
        }
    }
    
    // Convert from Array of Structures to Structure of Arrays (3D)
    void convert_aos_to_soa_3d(const CudaPOI3D* aos, POI3D_SoA& soa, int count)
    {
        for (int i = 0; i < count; i++)
        {
            // Position
            soa.x[i] = aos[i].x;
            soa.y[i] = aos[i].y;
            soa.z[i] = aos[i].z;
            
            // Deformation
            soa.u[i] = aos[i].deformation.u;
            soa.ux[i] = aos[i].deformation.ux;
            soa.uy[i] = aos[i].deformation.uy;
            soa.uz[i] = aos[i].deformation.uz;
            soa.v[i] = aos[i].deformation.v;
            soa.vx[i] = aos[i].deformation.vx;
            soa.vy[i] = aos[i].deformation.vy;
            soa.vz[i] = aos[i].deformation.vz;
            soa.w[i] = aos[i].deformation.w;
            soa.wx[i] = aos[i].deformation.wx;
            soa.wy[i] = aos[i].deformation.wy;
            soa.wz[i] = aos[i].deformation.wz;
            
            // Result
            soa.u0[i] = aos[i].result.u0;
            soa.v0[i] = aos[i].result.v0;
            soa.w0[i] = aos[i].result.w0;
            soa.zncc[i] = aos[i].result.zncc;
            soa.iteration[i] = aos[i].result.iteration;
            soa.convergence[i] = aos[i].result.convergence;
            soa.feature[i] = aos[i].result.feature;
            
            // Strain
            soa.exx[i] = aos[i].strain.exx;
            soa.eyy[i] = aos[i].strain.eyy;
            soa.ezz[i] = aos[i].strain.ezz;
            soa.exy[i] = aos[i].strain.exy;
            soa.eyz[i] = aos[i].strain.eyz;
            soa.ezx[i] = aos[i].strain.ezx;
            
            // Subset radius
            soa.subset_radius_x[i] = aos[i].subset_radius_x;
            soa.subset_radius_y[i] = aos[i].subset_radius_y;
            soa.subset_radius_z[i] = aos[i].subset_radius_z;
        }
    }
    
    // Convert from Structure of Arrays to Array of Structures (3D)
    void convert_soa_to_aos_3d(const POI3D_SoA& soa, CudaPOI3D* aos, int count)
    {
        for (int i = 0; i < count; i++)
        {
            // Position
            aos[i].x = soa.x[i];
            aos[i].y = soa.y[i];
            aos[i].z = soa.z[i];
            
            // Deformation
            aos[i].deformation.u = soa.u[i];
            aos[i].deformation.ux = soa.ux[i];
            aos[i].deformation.uy = soa.uy[i];
            aos[i].deformation.uz = soa.uz[i];
            aos[i].deformation.v = soa.v[i];
            aos[i].deformation.vx = soa.vx[i];
            aos[i].deformation.vy = soa.vy[i];
            aos[i].deformation.vz = soa.vz[i];
            aos[i].deformation.w = soa.w[i];
            aos[i].deformation.wx = soa.wx[i];
            aos[i].deformation.wy = soa.wy[i];
            aos[i].deformation.wz = soa.wz[i];
            
            // Result
            aos[i].result.u0 = soa.u0[i];
            aos[i].result.v0 = soa.v0[i];
            aos[i].result.w0 = soa.w0[i];
            aos[i].result.zncc = soa.zncc[i];
            aos[i].result.iteration = soa.iteration[i];
            aos[i].result.convergence = soa.convergence[i];
            aos[i].result.feature = soa.feature[i];
            
            // Strain
            aos[i].strain.exx = soa.exx[i];
            aos[i].strain.eyy = soa.eyy[i];
            aos[i].strain.ezz = soa.ezz[i];
            aos[i].strain.exy = soa.exy[i];
            aos[i].strain.eyz = soa.eyz[i];
            aos[i].strain.ezx = soa.ezx[i];
            
            // Subset radius
            aos[i].subset_radius_x = soa.subset_radius_x[i];
            aos[i].subset_radius_y = soa.subset_radius_y[i];
            aos[i].subset_radius_z = soa.subset_radius_z[i];
        }
    }
    
    // Copy POI2D_SoA from host to device
    void copy_poi2d_soa_to_device(const POI2D_SoA& host_soa, POI2D_SoA& device_soa, cudaStream_t stream)
    {
        size_t size = host_soa.count * sizeof(float);
        
        CUDA_CHECK(cudaMemcpyAsync(device_soa.x, host_soa.x, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.y, host_soa.y, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.u, host_soa.u, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.ux, host_soa.ux, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.uy, host_soa.uy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.uxx, host_soa.uxx, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.uxy, host_soa.uxy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.uyy, host_soa.uyy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.v, host_soa.v, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.vx, host_soa.vx, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.vy, host_soa.vy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.vxx, host_soa.vxx, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.vxy, host_soa.vxy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.vyy, host_soa.vyy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.u0, host_soa.u0, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.v0, host_soa.v0, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.zncc, host_soa.zncc, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.iteration, host_soa.iteration, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.convergence, host_soa.convergence, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.feature, host_soa.feature, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.exx, host_soa.exx, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.eyy, host_soa.eyy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.exy, host_soa.exy, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.subset_radius_x, host_soa.subset_radius_x, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_soa.subset_radius_y, host_soa.subset_radius_y, size, cudaMemcpyHostToDevice, stream));
    }
    
    // Copy POI2D_SoA from device to host
    void copy_poi2d_soa_to_host(const POI2D_SoA& device_soa, POI2D_SoA& host_soa, cudaStream_t stream)
    {
        size_t size = device_soa.count * sizeof(float);
        
        cudaMemcpyAsync(host_soa.x, device_soa.x, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.y, device_soa.y, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.u, device_soa.u, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.ux, device_soa.ux, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uy, device_soa.uy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uxx, device_soa.uxx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uxy, device_soa.uxy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uyy, device_soa.uyy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.v, device_soa.v, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vx, device_soa.vx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vy, device_soa.vy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vxx, device_soa.vxx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vxy, device_soa.vxy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vyy, device_soa.vyy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.u0, device_soa.u0, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.v0, device_soa.v0, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.zncc, device_soa.zncc, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.iteration, device_soa.iteration, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.convergence, device_soa.convergence, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.feature, device_soa.feature, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.exx, device_soa.exx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.eyy, device_soa.eyy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.exy, device_soa.exy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.subset_radius_x, device_soa.subset_radius_x, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.subset_radius_y, device_soa.subset_radius_y, size, cudaMemcpyDeviceToHost, stream);
    }
    
    // Copy POI3D_SoA from host to device
    void copy_poi3d_soa_to_device(const POI3D_SoA& host_soa, POI3D_SoA& device_soa, cudaStream_t stream)
    {
        size_t size = host_soa.count * sizeof(float);
        
        cudaMemcpyAsync(device_soa.x, host_soa.x, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.y, host_soa.y, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.z, host_soa.z, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.u, host_soa.u, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.ux, host_soa.ux, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.uy, host_soa.uy, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.uz, host_soa.uz, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.v, host_soa.v, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.vx, host_soa.vx, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.vy, host_soa.vy, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.vz, host_soa.vz, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.w, host_soa.w, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.wx, host_soa.wx, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.wy, host_soa.wy, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.wz, host_soa.wz, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.u0, host_soa.u0, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.v0, host_soa.v0, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.w0, host_soa.w0, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.zncc, host_soa.zncc, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.iteration, host_soa.iteration, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.convergence, host_soa.convergence, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.feature, host_soa.feature, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.exx, host_soa.exx, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.eyy, host_soa.eyy, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.ezz, host_soa.ezz, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.exy, host_soa.exy, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.eyz, host_soa.eyz, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.ezx, host_soa.ezx, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.subset_radius_x, host_soa.subset_radius_x, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.subset_radius_y, host_soa.subset_radius_y, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_soa.subset_radius_z, host_soa.subset_radius_z, size, cudaMemcpyHostToDevice, stream);
    }
    
    // Copy POI3D_SoA from device to host
    void copy_poi3d_soa_to_host(const POI3D_SoA& device_soa, POI3D_SoA& host_soa, cudaStream_t stream)
    {
        size_t size = device_soa.count * sizeof(float);
        
        cudaMemcpyAsync(host_soa.x, device_soa.x, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.y, device_soa.y, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.z, device_soa.z, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.u, device_soa.u, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.ux, device_soa.ux, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uy, device_soa.uy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.uz, device_soa.uz, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.v, device_soa.v, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vx, device_soa.vx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vy, device_soa.vy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.vz, device_soa.vz, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.w, device_soa.w, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.wx, device_soa.wx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.wy, device_soa.wy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.wz, device_soa.wz, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.u0, device_soa.u0, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.v0, device_soa.v0, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.w0, device_soa.w0, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.zncc, device_soa.zncc, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.iteration, device_soa.iteration, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.convergence, device_soa.convergence, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.feature, device_soa.feature, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.exx, device_soa.exx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.eyy, device_soa.eyy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.ezz, device_soa.ezz, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.exy, device_soa.exy, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.eyz, device_soa.eyz, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.ezx, device_soa.ezx, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.subset_radius_x, device_soa.subset_radius_x, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.subset_radius_y, device_soa.subset_radius_y, size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_soa.subset_radius_z, device_soa.subset_radius_z, size, cudaMemcpyDeviceToHost, stream);
    }
    
} // namespace StudyCorr_GPU
