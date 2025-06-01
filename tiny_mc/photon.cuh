#include "xorshift32.cuh"

#pragma once


/**
* @brief Simula la propagación de múltiples fotones en un medio dispersivo y absorbente.
*
* @param heats             Arreglo que acumula la energía absorbida por shell.
* @param heats_squared     Arreglo que acumula el cuadrado de la energía absorbida (para varianza).
* @param rng               Puntero al generador de números aleatorios (Xorshift32).
*/
__device__ __forceinline__ void photon(float* heats, float* heats_squared, Xorshift32* rng)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        float t = -logf(xorshift32_norm(rng));
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp;
        if (shell >= SHELLS) shell = SHELLS - 1;

        float absorb = (1.0f - albedo) * weight;
        atomicAdd(&heats[shell], absorb);
        atomicAdd(&heats_squared[shell], absorb * absorb);

        weight *= albedo;

        float xi1, xi2;
        do {
            xi1 = 2.0f * xorshift32_norm(rng) - 1.0f;
            xi2 = 2.0f * xorshift32_norm(rng) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        u = 2.0f * t - 1.0f;
        float sqrt_term = sqrtf((1.0f - u * u) / t);
        v = xi1 * sqrt_term;
        w = xi2 * sqrt_term;

        if (weight < 0.001f) {
            if (xorshift32_norm(rng) > 0.1f) break;
            weight /= 0.1f;
        }
    }
}

