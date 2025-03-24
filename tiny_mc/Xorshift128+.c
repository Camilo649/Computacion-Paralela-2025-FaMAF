#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint64_t state[2];
} Xorshift128Plus;

void xorshift128plus_init(Xorshift128Plus* rng, uint64_t seed1, uint64_t seed2)
{
    rng->state[0] = seed1;
    rng->state[1] = seed2;
}

uint64_t xorshift128plus_next(Xorshift128Plus* rng)
{
    uint64_t s1 = rng->state[0];
    uint64_t s0 = rng->state[1];
    rng->state[0] = s0;
    s1 ^= s1 << 23; // Desplazamiento y XOR
    rng->state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26); // Más operaciones
    return rng->state[1] + s0;
}

// Generación de un número aleatorio en el rango [0, 1) usando Xorshift128+
float xorshift128plus_randf(Xorshift128Plus* rng)
{
    return (float)xorshift128plus_next(rng) / (float)UINT64_MAX;
}
