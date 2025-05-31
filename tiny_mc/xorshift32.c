#include "xorshift32.h"

void xorshift32_init(Xorshift32 * restrict rng, uint32_t seed) {
    rng->state = seed;
}

uint32_t xorshift32(Xorshift32 * restrict rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

float xorshift32_norm(Xorshift32 * restrict rng) {
    return xorshift32(rng) / (float)UINT32_MAX;
}
