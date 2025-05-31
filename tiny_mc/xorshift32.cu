#pragma once

#include "xorshift32.cuh"


__device__ void xorshift32_init(Xorshift32* rng, uint32_t seed) {
    rng->state = seed;
}

__device__ float xorshift32_norm(Xorshift32* rng) {
    rng->state ^= rng->state << 13;
    rng->state ^= rng->state >> 17;
    rng->state ^= rng->state << 5;
    return return rng->state * (1.0f / 4294967296.0f);
}
