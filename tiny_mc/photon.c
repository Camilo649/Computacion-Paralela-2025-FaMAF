#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "xorshift32.h"
#include "params.h"
#include "photon.h"
#include "xorshift32.h"
#include "params.h"

void photon8(Xorshift32* restrict rng,
             float* restrict heats,
             float* restrict heats_squared,
             size_t photon_loops)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);

    const float mu_total = MU_S + MU_A;
    const float albedo = MU_S / mu_total;
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / mu_total;

    // Estado de los 8 fotones
    __m256 x = zero, y = zero, z = zero;
    __m256 u = zero, v = zero, w = one;
    __m256 weight = one;

     size_t total_remaining = photon_loops * 8;

    while (total_remaining > 0) {

        /* move */
         __m256 rand_vals = xorshift32_randf8(rng);
        __m256 t = _mm256_log_ps(rand_vals);
        t = _mm256_mul_ps(t, _mm256_set1_ps(-1.0f));
        x = _mm256_fmadd_ps(t, u, x);
        y = _mm256_fmadd_ps(t, v, y);
        z = _mm256_fmadd_ps(t, w, z);

        /* absorb */
        __m256 r2 = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_add_ps(_mm256_mul_ps(y, y), _mm256_mul_ps(z, z)));
        __m256 sqrt_r2 = _mm256_mul_ps(r2, _mm256_rsqrt_ps(r2));
        __m256 shell = _mm256_mul_ps(sqrt_r2, _mm256_set1_ps(shells_per_mfp));
        shell = _mm256_min_ps(shell, _mm256_set1_ps(SHELLS - 1));
        __m256 absorbed = _mm256_mul_ps(_mm256_set1_ps(1.0f - albedo), weight);
        for (int i = 0; i < 8; i++) {
            int shell_idx = (int)((float*)&shell)[i];
            heats[shell_idx] += ((float*)&absorbed)[i];
            heats_squared[shell_idx] += ((float*)&absorbed)[i] * ((float*)&absorbed)[i];
        }
        weight = _mm256_mul_ps(weight, _mm256_set1_ps(albedo));

        __m256 xi1_valid = zero;
        __m256 xi2_valid = zero;
        __m256 t_valid   = one;
        __m256 done_mask = zero;

        do {
            __m256 xi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng)), _mm256_set1_ps(1.0f));
            __m256 xi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng)), _mm256_set1_ps(1.0f));
            __m256 t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
        
            __m256 new_valid = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_LT_OS);
            __m256 new_data_mask = _mm256_andnot_ps(done_mask, new_valid);
        
            xi1_valid = _mm256_blendv_ps(xi1_valid, xi1, new_data_mask);
            xi2_valid = _mm256_blendv_ps(xi2_valid, xi2, new_data_mask);
            t_valid   = _mm256_blendv_ps(t_valid,   t,   new_data_mask);
        
            done_mask = _mm256_or_ps(done_mask, new_valid);
        } while (_mm256_movemask_ps(done_mask) != 0xFF);

        __m256 sqrt_factor = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(xi1_valid, xi1_valid)), _mm256_rsqrt_ps(t_valid));
        u = _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(xi1_valid, sqrt_factor));
        v = _mm256_mul_ps(xi1_valid, sqrt_factor);
        w = _mm256_mul_ps(xi2_valid, sqrt_factor);

        // DEATH + RECYCLE
        __m256 dead_mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
        int mask = _mm256_movemask_ps(dead_mask);
        for (int i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                if (total_remaining > 0) {
                    total_remaining--;
                    ((float*)&x)[i] = 0.0f;
                    ((float*)&y)[i] = 0.0f;
                    ((float*)&z)[i] = 0.0f;
                    ((float*)&u)[i] = 0.0f;
                    ((float*)&v)[i] = 0.0f;
                    ((float*)&w)[i] = 1.0f;
                    ((float*)&weight)[i] = 1.0f;
                } else {
                    // Desactivar la lane
                    ((float*)&weight)[i] = 0.0f;
                }
            }
        }
    }
}


