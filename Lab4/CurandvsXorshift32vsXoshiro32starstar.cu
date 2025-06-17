#include <stdio.h>
#include <curand_kernel.h>
#include <stdint.h>

#define N (2<<24)
#define THREADS_PER_BLOCK 256

// -----------------------------
// RNG 1: Xorshift32 simple
// -----------------------------
__device__ uint32_t d_state_xs[N];

__global__ void init_xorshift32(uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_state_xs[idx] = seed + idx * 12345;
    }
}

__device__ uint32_t xorshift32(int idx) {
    uint32_t s = d_state_xs[idx];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    d_state_xs[idx] = s;
    return s;
}

__device__ float xorshift32f(int idx) {
    return xorshift32(idx) * (1.0f / 4294967296.0f);
}

__global__ void generate_xorshift32(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = xorshift32f(idx);
    }
}

// -----------------------------
// RNG 2: Xoshiro32**
// -----------------------------
__device__ uint32_t d_state_xoshiro0[N];
__device__ uint32_t d_state_xoshiro1[N];

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__global__ void init_xoshiro(uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t z = seed + idx * 0x9E3779B9;
        z = (z ^ (z >> 16)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        d_state_xoshiro0[idx] = z ^ (z >> 16);

        z += 0x9E3779B9;
        z = (z ^ (z >> 16)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        d_state_xoshiro1[idx] = z ^ (z >> 16);

        if (d_state_xoshiro0[idx] == 0 && d_state_xoshiro1[idx] == 0)
            d_state_xoshiro0[idx] = 1;
    }
}

__device__ uint32_t xoshiro32starstar(int idx) {
    uint32_t s0 = d_state_xoshiro0[idx];
    uint32_t s1 = d_state_xoshiro1[idx];

    uint32_t result = rotl32(s0 * 5, 7) * 9;

    s1 ^= s0;
    d_state_xoshiro0[idx] = rotl32(s0, 26) ^ s1 ^ (s1 << 9);
    d_state_xoshiro1[idx] = rotl32(s1, 13);

    return result;
}

__device__ float xoshiro32f(int idx) {
    return xoshiro32starstar(idx) * (1.0f / 4294967296.0f);
}

__global__ void generate_xoshiro(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = xoshiro32f(idx);
    }
}

// -----------------------------
// RNG 3: cuRAND XORWOW
// -----------------------------
__global__ void init_curand(curandStateXORWOW_t* states, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void generate_curand(curandStateXORWOW_t* states, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = curand_uniform(&states[idx]);
    }
}

// -----------------------------
// MAIN
// -----------------------------
int main() {
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Xorshift32 ---
    init_xorshift32<<<blocks, THREADS_PER_BLOCK>>>(1234);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    generate_xorshift32<<<blocks, THREADS_PER_BLOCK>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed1 = 0;
    cudaEventElapsedTime(&elapsed1, start, stop);
    printf("Xorshift32: Tiempo para %d números: %f ms\n", N, elapsed1);

    // --- Xoshiro32** ---
    init_xoshiro<<<blocks, THREADS_PER_BLOCK>>>(5678);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    generate_xoshiro<<<blocks, THREADS_PER_BLOCK>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed2 = 0;
    cudaEventElapsedTime(&elapsed2, start, stop);
    printf("Xoshiro32**: Tiempo para %d números: %f ms\n", N, elapsed2);

    // --- cuRAND XORWOW ---
    curandStateXORWOW_t* d_states;
    cudaMalloc(&d_states, N * sizeof(curandStateXORWOW_t));
    init_curand<<<blocks, THREADS_PER_BLOCK>>>(d_states, 9012);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    generate_curand<<<blocks, THREADS_PER_BLOCK>>>(d_states, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed3 = 0;
    cudaEventElapsedTime(&elapsed3, start, stop);
    printf("cuRAND XORWOW: Tiempo para %d números: %f ms\n", N, elapsed3);

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
