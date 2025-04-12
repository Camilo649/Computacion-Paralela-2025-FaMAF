#include <math.h>

#include "xorshift32.h"
#include "params.h"

void photon(Xorshift32* rng, float* heats, float* heats_squared)
{
    const float mu_total = MU_S + MU_A;
    const float albedo = MU_S / mu_total;
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / mu_total;

    const float one_minus_albedo = 1.0f - albedo;
    const float one_minus_albedo_sq = one_minus_albedo * one_minus_albedo;
    const float roulette_threshold = 0.001f;
    const float roulette_survival_prob = 0.1f;
    const float roulette_weight_boost = 1.0f / roulette_survival_prob;

    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f; 
    float v = 0.0f; 
    float w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        float t = -logf(xorshift32_randf(rng));  // move
        x = fmaf(t, u, x);
        y = fmaf(t, v, y);
        z = fmaf(t, w, z);

        float r2 = x * x + y * y + z * z;
        unsigned int shell = sqrtf(r2) * shells_per_mfp;  // absorb

        if (shell > SHELLS - 1)
            shell = SHELLS - 1;

        float absorbed = one_minus_albedo * weight;
        heats[shell] += absorbed;
        heats_squared[shell] += one_minus_albedo_sq * weight * weight;
        weight *= albedo;

        // New direction, rejection method
        float xi1, xi2;
        do {
            xi1 = 2.0f * xorshift32_randf(rng) - 1.0f;
            xi2 = 2.0f * xorshift32_randf(rng) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (t >= 1.0f);
        u = 2.0f * t - 1.0f;
        float sqrt_factor = sqrtf((1.0f - u * u) / t);
        v = xi1 * sqrt_factor;
        w = xi2 * sqrt_factor;

        // roulette
        if (weight < roulette_threshold) {
            if (xorshift32_randf(rng) > roulette_survival_prob)
                break;
            weight *= roulette_weight_boost;
        }
    }
}
