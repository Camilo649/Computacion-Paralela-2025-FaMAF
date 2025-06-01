#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "params.h"
#include "photon.cuh"
#include "xorshift32.cuh"


__global__ void simulate_kernel(float* heats, float* heats_squared)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PHOTONS) return;
    Xorshift32 rng;
    xorshift32_init(&rng, SEED ^ (threadIdx.x * 0x9E3779B9u) ^ (blockIdx.x * 0x85EBCA6Bu) ^ (blockDim.x * 0xC2B2AE35u));
    photon(heats, heats_squared, &rng);
}


int main()
{
    const int size = SHELLS * sizeof(float);

    float* d_heats;
    float* d_heats_squared;
    cudaMalloc(&d_heats, size);
    cudaMalloc(&d_heats_squared, size);
    cudaMemset(d_heats, 0, size);
    cudaMemset(d_heats_squared, 0, size);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Reservamos toda la RAM libre del GPU para que nadie mas la usea MUAJAJAJAJ
    void* gobble;
    cudaMalloc(&gobble, free_mem - (16 * 1024 * 1024)); // dejamos 16 MiB libres por seguridad

    float elapsed_time;
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e1);

    simulate_kernel<<<PHOTONS/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_heats, d_heats_squared);
    cudaDeviceSynchronize();

    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&elapsed_time, e1, e2);

    float* heats = (float*)malloc(size);
    cudaMemcpy(heats, d_heats, size, cudaMemcpyDeviceToHost);
    float* heats_squared = (float*)malloc(size);
    cudaMemcpy(heats_squared, d_heats_squared, size, cudaMemcpyDeviceToHost);

    printf("%f\n", PHOTONS / (1000.0f * elapsed_time));

    cudaFree(d_heats);
    cudaFree(d_heats_squared);
    free(heats);
    free(heats_squared);

    return 0;
}
