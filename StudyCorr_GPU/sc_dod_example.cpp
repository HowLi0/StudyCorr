/*
 * Example demonstrating the use of DOD (Data-Oriented Design) interfaces
 * for high-performance GPU-accelerated DIC computation
 */

#include "StudyCorr_GPU.h"
#include <vector>
#include <iostream>
#include <chrono>

using namespace StudyCorr_GPU;

// Example: 2D ICGN computation using DOD (Structure of Arrays) interface
void example_icgn2d_dod()
{
    std::cout << "=== DOD ICGN2D Example ===" << std::endl;
    
    // 1. Prepare input images (example dimensions)
    int width = 1024;
    int height = 1024;
    std::vector<float> ref_image(width * height, 100.0f);
    std::vector<float> tar_image(width * height, 100.0f);
    
    // In a real application, load actual images here
    // For example: cv::Mat ref_img = cv::imread("reference.png", cv::IMREAD_GRAYSCALE);
    
    // 2. Prepare POIs (Points of Interest) - traditional AoS format
    int num_pois = 1000;
    std::vector<CudaPOI2D> pois_aos(num_pois);
    
    // Initialize POIs with positions in a grid
    int grid_size = static_cast<int>(sqrt(num_pois));
    int step = 50;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int idx = i * grid_size + j;
            if (idx >= num_pois) break;
            
            pois_aos[idx].x = 100 + j * step;
            pois_aos[idx].y = 100 + i * step;
            pois_aos[idx].subset_radius_x = 21;
            pois_aos[idx].subset_radius_y = 21;
            
            // Initialize deformation to zero
            pois_aos[idx].deformation.u = 0.0f;
            pois_aos[idx].deformation.v = 0.0f;
            pois_aos[idx].deformation.ux = 0.0f;
            pois_aos[idx].deformation.uy = 0.0f;
            pois_aos[idx].deformation.vx = 0.0f;
            pois_aos[idx].deformation.vy = 0.0f;
        }
    }
    
    // 3. Convert AoS to SoA (Data-Oriented Design format)
    std::cout << "Converting to SoA format..." << std::endl;
    POI2D_SoA host_soa, device_soa;
    allocate_poi2d_soa_host(host_soa, num_pois);
    allocate_poi2d_soa_device(device_soa, num_pois);
    convert_aos_to_soa_2d(pois_aos.data(), host_soa, num_pois);
    
    // 4. Transfer data to GPU
    std::cout << "Transferring data to GPU..." << std::endl;
    auto start_transfer = std::chrono::high_resolution_clock::now();
    copy_poi2d_soa_to_device(host_soa, device_soa);
    cudaDeviceSynchronize();
    auto end_transfer = std::chrono::high_resolution_clock::now();
    auto transfer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_transfer - start_transfer).count();
    std::cout << "Transfer time: " << transfer_time << " ms" << std::endl;
    
    // 5. Setup ICGN parameters
    ICGNParam param;
    param.subsetRadius = 21;
    param.convergenceThreshold = 0.001;
    param.maxIterations = 10;
    
    // 6. Create DOD ICGN processor and prepare
    std::cout << "Preparing ICGN computation..." << std::endl;
    ICGN2D1BatchGpuDOD icgn_dod;
    icgn_dod.prepare_cuda(ref_image.data(), tar_image.data(), height, width, param);
    
    // 7. Perform computation
    std::cout << "Computing ICGN (1st order, " << num_pois << " POIs)..." << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();
    icgn_dod.compute_batch_cuda(device_soa);
    cudaDeviceSynchronize();
    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "Computation time: " << compute_time << " ms" << std::endl;
    
    // 8. Transfer results back to host
    std::cout << "Transferring results back to host..." << std::endl;
    copy_poi2d_soa_to_host(device_soa, host_soa);
    cudaDeviceSynchronize();
    
    // 9. Convert back to AoS format if needed
    convert_soa_to_aos_2d(host_soa, pois_aos.data(), num_pois);
    
    // 10. Check results
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < std::min(5, num_pois); i++) {
        std::cout << "  POI " << i << ": pos=(" << pois_aos[i].x << "," << pois_aos[i].y << ")"
                  << " u=" << pois_aos[i].deformation.u 
                  << " v=" << pois_aos[i].deformation.v
                  << " zncc=" << pois_aos[i].result.zncc
                  << " iter=" << pois_aos[i].result.iteration << std::endl;
    }
    
    // 11. Cleanup
    icgn_dod.release_cuda();
    free_poi2d_soa(host_soa);
    free_poi2d_soa(device_soa);
    
    std::cout << "=== Example completed ===" << std::endl << std::endl;
}

// Example: Batch processing with asynchronous streams
void example_batch_processing_async()
{
    std::cout << "=== DOD Batch Processing with Async Streams ===" << std::endl;
    
    int width = 1024;
    int height = 1024;
    std::vector<float> ref_image(width * height, 100.0f);
    std::vector<float> tar_image(width * height, 100.0f);
    
    // Create multiple batches
    const int num_batches = 4;
    const int batch_size = 256;
    std::vector<POI2D_SoA> host_batches(num_batches);
    std::vector<POI2D_SoA> device_batches(num_batches);
    
    // Initialize batches
    for (int b = 0; b < num_batches; b++) {
        allocate_poi2d_soa_host(host_batches[b], batch_size);
        allocate_poi2d_soa_device(device_batches[b], batch_size);
        
        // Initialize POI positions
        for (int i = 0; i < batch_size; i++) {
            host_batches[b].x[i] = 100 + (i % 16) * 50;
            host_batches[b].y[i] = 100 + (i / 16) * 50;
            host_batches[b].subset_radius_x[i] = 21;
            host_batches[b].subset_radius_y[i] = 21;
        }
    }
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Setup ICGN
    ICGNParam param;
    param.subsetRadius = 21;
    param.convergenceThreshold = 0.001;
    param.maxIterations = 10;
    
    ICGN2D1BatchGpuDOD icgn_dod;
    icgn_dod.prepare_cuda(ref_image.data(), tar_image.data(), height, width, param, stream);
    
    std::cout << "Processing " << num_batches << " batches asynchronously..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pipeline: transfer->compute->transfer for all batches
    for (int b = 0; b < num_batches; b++) {
        copy_poi2d_soa_to_device(host_batches[b], device_batches[b], stream);
        icgn_dod.compute_batch_cuda(device_batches[b], stream);
        copy_poi2d_soa_to_host(device_batches[b], host_batches[b], stream);
    }
    
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Total processing time: " << total_time << " ms" << std::endl;
    std::cout << "Average per batch: " << (total_time / num_batches) << " ms" << std::endl;
    
    // Cleanup
    cudaStreamDestroy(stream);
    icgn_dod.release_cuda();
    for (int b = 0; b < num_batches; b++) {
        free_poi2d_soa(host_batches[b]);
        free_poi2d_soa(device_batches[b]);
    }
    
    std::cout << "=== Batch processing completed ===" << std::endl << std::endl;
}

// Comparison: AoS vs SoA performance
void example_performance_comparison()
{
    std::cout << "=== Performance Comparison: AoS vs SoA ===" << std::endl;
    
    int width = 1024;
    int height = 1024;
    int num_pois = 1000;
    
    std::vector<float> ref_image(width * height, 100.0f);
    std::vector<float> tar_image(width * height, 100.0f);
    std::vector<CudaPOI2D> pois_aos(num_pois);
    
    // Initialize POIs
    for (int i = 0; i < num_pois; i++) {
        pois_aos[i].x = 100 + (i % 32) * 30;
        pois_aos[i].y = 100 + (i / 32) * 30;
        pois_aos[i].subset_radius_x = 21;
        pois_aos[i].subset_radius_y = 21;
    }
    
    ICGNParam param;
    param.subsetRadius = 21;
    param.convergenceThreshold = 0.001;
    param.maxIterations = 10;
    
    // Test traditional AoS approach
    std::cout << "Testing traditional AoS approach..." << std::endl;
    ICGN2D1BatchGpu icgn_aos;
    icgn_aos.prepare_cuda(ref_image.data(), tar_image.data(), height, width, param);
    
    CudaPOI2D* d_pois_aos;
    cudaMalloc(&d_pois_aos, num_pois * sizeof(CudaPOI2D));
    cudaMemcpy(d_pois_aos, pois_aos.data(), num_pois * sizeof(CudaPOI2D), cudaMemcpyHostToDevice);
    
    auto start_aos = std::chrono::high_resolution_clock::now();
    icgn_aos.compute_batch_cuda(d_pois_aos, num_pois);
    cudaDeviceSynchronize();
    auto end_aos = std::chrono::high_resolution_clock::now();
    auto time_aos = std::chrono::duration_cast<std::chrono::microseconds>(end_aos - start_aos).count();
    
    cudaMemcpy(pois_aos.data(), d_pois_aos, num_pois * sizeof(CudaPOI2D), cudaMemcpyDeviceToHost);
    cudaFree(d_pois_aos);
    icgn_aos.release_cuda();
    
    std::cout << "AoS time: " << time_aos / 1000.0 << " ms" << std::endl;
    
    // Test DOD SoA approach
    std::cout << "Testing DOD SoA approach..." << std::endl;
    POI2D_SoA host_soa, device_soa;
    allocate_poi2d_soa_host(host_soa, num_pois);
    allocate_poi2d_soa_device(device_soa, num_pois);
    convert_aos_to_soa_2d(pois_aos.data(), host_soa, num_pois);
    
    ICGN2D1BatchGpuDOD icgn_dod;
    icgn_dod.prepare_cuda(ref_image.data(), tar_image.data(), height, width, param);
    
    copy_poi2d_soa_to_device(host_soa, device_soa);
    
    auto start_soa = std::chrono::high_resolution_clock::now();
    icgn_dod.compute_batch_cuda(device_soa);
    cudaDeviceSynchronize();
    auto end_soa = std::chrono::high_resolution_clock::now();
    auto time_soa = std::chrono::duration_cast<std::chrono::microseconds>(end_soa - start_soa).count();
    
    copy_poi2d_soa_to_host(device_soa, host_soa);
    
    std::cout << "SoA time: " << time_soa / 1000.0 << " ms" << std::endl;
    std::cout << "Speedup: " << (double)time_aos / time_soa << "x" << std::endl;
    
    // Cleanup
    icgn_dod.release_cuda();
    free_poi2d_soa(host_soa);
    free_poi2d_soa(device_soa);
    
    std::cout << "=== Performance comparison completed ===" << std::endl << std::endl;
}

int main()
{
    std::cout << "StudyCorr DOD (Data-Oriented Design) Examples" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;
    
    // Run examples
    example_icgn2d_dod();
    example_batch_processing_async();
    example_performance_comparison();
    
    std::cout << "All examples completed successfully!" << std::endl;
    return 0;
}
