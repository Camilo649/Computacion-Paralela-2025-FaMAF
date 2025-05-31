#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "params.h"
#include "photon.cuh"
#include "xorshift32.cuh"


__global__ void simulate_kernel(float* heats, float* heats_squared, unsigned int photons, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= photons) return;

    Xorshift32 rng;
    xorshift32_init(&rng, seed ^ tid);  // semilla única por hilo

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

    double elapsed_time;
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e1);

    simulate_kernel<<<PHOTONS/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_heats, d_heats_squared, num_photons, SEED);
    cudaDeviceSynchronize();

    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&elapsed_time, e1, e2);

    float* heats = (float*)malloc(size);
    cudaMemcpy(heats, d_heats, size, cudaMemcpyDeviceToHost);

    printf("%f\n", 1e-3 * PHOTONS / elapsed_time); // Kfotones / ms

    cudaFree(d_heats);
    cudaFree(d_heats_squared);
    free(heats);

    return 0;
}
