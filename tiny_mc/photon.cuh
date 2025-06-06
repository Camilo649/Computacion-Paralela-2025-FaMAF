#include <cuda_runtime.h>

#include "xorshift32.cuh"

#pragma once

/**
* @brief Simula la propagación de múltiples fotones en un medio dispersivo y absorbente.
*
* @param heats             Arreglo que acumula la energía absorbida por shell.
* @param heats_squared     Arreglo que acumula el cuadrado de la energía absorbida (para varianza).
* @param rng               Puntero al generador de números aleatorios (Xorshift32).
*/
__device__ __forceinline__ void photon(float* __restrict__ heats, float* __restrict__ heats_squared, Xorshift32* __restrict__ rng)
{
    unsigned long photons = PHOTONS_PER_THREAD;
    const float inv_mu = __frcp_rn(MU_A + MU_S);
    const float albedo = __fmul_rn(MU_S, inv_mu);
    const float shells_per_mfp = __fdiv_rn(1e4f, __fmul_rn(MICRONS_PER_SHELL, __fadd_rn(MU_A, MU_S)));

    while (photons-- > 0) {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float u = 0.0f, v = 0.0f, w = 1.0f;
        float weight = 1.0f;

        for (;;) {
            float t = -__logf(xorshift32_norm(rng));
            x = __fmaf_rn(t, u, x);
            y = __fmaf_rn(t, v, y);
            z = __fmaf_rn(t, w, z);

            float r2 = __fmaf_rn(x, x, __fmaf_rn(y, y, z * z));
            float rsqrt_r = __frsqrt_rn(r2);
            float r = __fmul_rn(r2, rsqrt_r);
            unsigned int shell = (unsigned int)(__fmul_rn(r, shells_per_mfp));
            if (shell >= SHELLS) shell = SHELLS - 1;

            float absorb = __fmul_rn(__fsub_rn(1.0f, albedo), weight);
            atomicAdd(&heats[shell], absorb);
            atomicAdd(&heats_squared[shell], __fmul_rn(absorb, absorb));

            weight = __fmul_rn(weight, albedo);

            float xi1, xi2;
            do {
                xi1 = __fsub_rn(__fmul_rn(2.0f, xorshift32_norm(rng)), 1.0f);
                xi2 = __fsub_rn(__fmul_rn(2.0f, xorshift32_norm(rng)), 1.0f);
                t = __fmaf_rn(xi1, xi1, xi2 * xi2);
            } while (t >= 1.0f);

            u = __fsub_rn(__fmul_rn(2.0f, t), 1.0f);
            float one_minus_u2 = __fsub_rn(1.0f, u * u);
            float sqrt_term = __fmul_rn(one_minus_u2, __frsqrt_rn(t));
            v = __fmul_rn(xi1, sqrt_term);
            w = __fmul_rn(xi2, sqrt_term);

            if (weight < 0.001f) {
                if (xorshift32_norm(rng) > 0.1f) break;
                weight = __fmul_rn(weight, 10.0f);
            }
        }
    }
}

