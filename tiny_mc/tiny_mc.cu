#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "params.h"
#include "photon.cuh"
#include "xorshift32.cuh"


__global__ void simulate_kernel(float* __restrict__ heats, float* __restrict__ heats_squared)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= PHOTONS) return;
    int btid = threadIdx.x;

    // Fase 1: Incialización
    __shared__ float heats_local[SHELLS];
    __shared__ float heats_squared_local[SHELLS];
    Xorshift32 rng;
    xorshift32_init(&rng, SEED ^ (btid * 0x9E3779B9u) ^ (blockIdx.x * 0x85EBCA6Bu) ^ (blockDim.x * 0xC2B2AE35u));
    if (btid < SHELLS) { // OJO! Solo funciona si THREADS_PER_BLOCK >= SHELLS
        heats_local[btid] = 0.0f;
        heats_squared_local[btid] = 0.0f;
    }
    __syncthreads();

    // Fase 2: Cómputo
    photon(heats_local, heats_squared_local, &rng);
    __syncthreads();

    // Fase 3: Acumulación
    if (btid < SHELLS) { // OJO! Solo funciona si THREADS_PER_BLOCK >= SHELLS
        atomicAdd(&heats[btid], heats_local[btid]);
        atomicAdd(&heats_squared[btid], heats_squared_local[btid]);
    }
}


int main() { 
    
    assert(THREADS_PER_BLOCK >= SHELLS && "THREADS_PER_BLOCK debe ser mayor o igual a SHELLS");

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
    //void* gobble;
    //cudaMalloc(&gobble, free_mem - (16 * 1024 * 1024)); // dejamos 16 MiB libres por seguridad

    float elapsed_time;
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e1);

    simulate_kernel<<<(PHOTONS/THREADS_PER_BLOCK)/PHOTONS_PER_THREAD, THREADS_PER_BLOCK>>>(d_heats, d_heats_squared);
    cudaDeviceSynchronize();

    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&elapsed_time, e1, e2);

    float* heats = (float*)malloc(size);
    cudaMemcpy(heats, d_heats, size, cudaMemcpyDeviceToHost);
    float* heats_squared = (float*)malloc(size);
    cudaMemcpy(heats_squared, d_heats_squared, size, cudaMemcpyDeviceToHost);

    printf("%f\n", PHOTONS / (1000.0f * elapsed_time));

    // printf("# Radius\tHeat\n");
    // printf("# [microns]\t[W/cm^3]\tError\n");
    // float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    // for (unsigned int i = 0; i < SHELLS - 1; ++i) {
    //     printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
    //            heats[i] / t / (i * i + i + 1.0 / 3.0),
    //            sqrt(heats_squared[i] - heats[i] * heats[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    // }
    // printf("# extra\t%12.5f\n", heats[SHELLS - 1] / PHOTONS);

    cudaFree(d_heats);
    cudaFree(d_heats_squared);
    free(heats);
    free(heats_squared);

    return 0;
}
