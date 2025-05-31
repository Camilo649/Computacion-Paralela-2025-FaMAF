#include <stdio.h>
#include <curand_kernel.h>
#include <stdint.h>

#define N (1024 * 1024)  // total números a generar
#define THREADS_PER_BLOCK 256

// Estado global para xorshift (un estado por hilo)
__device__ uint32_t d_state[N];

// Inicializa el estado xorshift para cada hilo (semilla arbitraria)
__global__ void init_xorshift(uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Semilla distinta para cada hilo, por ejemplo sumando idx
        d_state[idx] = seed + idx * 12345;
    }
}

// xorshift32 (versión device, cada hilo usa su propio estado)
__device__ uint32_t xorshift32_device(int idx) {
    uint32_t s = d_state[idx];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    d_state[idx] = s;
    return s;
}

__device__ float xorshift32_float_device(int idx) {
    return xorshift32_device(idx) / (float)UINT32_MAX;
}

// Kernel que genera números con xorshift32
__global__ void generate_xorshift(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = xorshift32_float_device(idx);
    }
}

// Kernel que inicializa cuRAND XORWOW
__global__ void init_curand(curandStateXORWOW_t* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel que genera números con cuRAND XORWOW
__global__ void generate_curand(curandStateXORWOW_t* states, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = curand_uniform(&states[idx]);
    }
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // --- Xorshift32 ---
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_xorshift<<<blocks, THREADS_PER_BLOCK>>>(1234);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    generate_xorshift<<<blocks, THREADS_PER_BLOCK>>>(d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsed_xorshift = 0;
    cudaEventElapsedTime(&elapsed_xorshift, start, stop);

    printf("Xorshift32: Tiempo para %d números: %f ms\n", N, elapsed_xorshift);

    // --- cuRAND XORWOW ---
    curandStateXORWOW_t *d_states;
    cudaMalloc(&d_states, N * sizeof(curandStateXORWOW_t));

    init_curand<<<blocks, THREADS_PER_BLOCK>>>(d_states, 1234);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    generate_curand<<<blocks, THREADS_PER_BLOCK>>>(d_states, d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsed_curand = 0;
    cudaEventElapsedTime(&elapsed_curand, start, stop);

    printf("cuRAND XORWOW: Tiempo para %d números: %f ms\n", N, elapsed_curand);

    // Limpieza
    cudaFree(d_out);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
