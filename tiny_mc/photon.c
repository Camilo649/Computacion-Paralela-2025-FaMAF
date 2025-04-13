#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "xorshift32.h"
#include "params.h"
#include "photon.h"

void photon8(Xorshift32* rng, Photons* p, float* heats, float* heats_squared, size_t index)
{
    __m256 x = _mm256_loadu_ps(&p->x[index]);
    __m256 y = _mm256_loadu_ps(&p->y[index]);
    __m256 z = _mm256_loadu_ps(&p->z[index]);
    __m256 u = _mm256_loadu_ps(&p->u[index]);
    __m256 v = _mm256_loadu_ps(&p->v[index]);
    __m256 w = _mm256_loadu_ps(&p->w[index]);
    __m256 weight = _mm256_loadu_ps(&p->weight[index]);

    const float mu_total = MU_S + MU_A;
    const float albedo = MU_S / mu_total;
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / mu_total;

    const float one_minus_albedo = 1.0f - albedo;
    const float roulette_threshold = 0.001f;
    const float roulette_survival_prob = 0.1f;
    const float roulette_weight_boost = 1.0f / roulette_survival_prob;

    /* launch */
    __m256 zero = _mm256_set1_ps(0.0f);   // Vector para inicializar en 0
    __m256 one = _mm256_set1_ps(1.0f);     // Vector para inicializar en 1
    
    _mm256_storeu_ps(&p->x[index], zero);
    _mm256_storeu_ps(&p->y[index], zero);
    _mm256_storeu_ps(&p->z[index], zero);
    _mm256_storeu_ps(&p->u[index], zero);
    _mm256_storeu_ps(&p->v[index], zero);
    _mm256_storeu_ps(&p->w[index], one); 
    _mm256_storeu_ps(&p->weight[index], one);

    for (;;) {
        __m256 mask_alive = _mm256_cmp_ps(weight, _mm256_set1_ps(0.0f), _CMP_GT_OS);
        if (_mm256_testz_si256(_mm256_castps_si256(mask_alive), _mm256_castps_si256(mask_alive)) != 0) {
            break;
        }



        /* move */
        __m256 rand_vals = xorshift32_randf8(rng);
        __m256 t = _mm256_setzero_ps();
        for (int i = 0; i < 8; i++) {
            float rand_val = ((float*)&rand_vals)[i];
            float log_val = -logf(rand_val);
            ((float*)&t)[i] = log_val;          
        }
        x = _mm256_fmadd_ps(t, u, x);
        y = _mm256_fmadd_ps(t, v, y);
        z = _mm256_fmadd_ps(t, w, z);

        /* absorb */
        __m256 r2 = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_add_ps(_mm256_mul_ps(y, y), _mm256_mul_ps(z, z)));
        __m256 sqrt_r2 = _mm256_sqrt_ps(r2);
        __m256 shell = _mm256_mul_ps(sqrt_r2, _mm256_set1_ps(shells_per_mfp));
        __m256 max_shell = _mm256_set1_ps(SHELLS - 1);
        shell = _mm256_min_ps(shell, max_shell);
        __m256 absorbed = _mm256_mul_ps(_mm256_set1_ps(one_minus_albedo), weight);
        for (int i = 0; i < 8; i++) {
            int shell_idx = (int)((float*)&shell)[i];
            if (_mm256_testz_si256(_mm256_castps_si256(mask_alive), _mm256_set1_epi32(1)) == 0) {
                heats[shell_idx] += ((float*)&absorbed)[i];
                heats_squared[shell_idx] += ((float*)&absorbed)[i] * ((float*)&absorbed)[i];
            }
        }
        weight = _mm256_mul_ps(weight, _mm256_set1_ps(albedo));

        /* New direction, rejection method */
        __m256 xi1 = _mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng));
        __m256 xi2 = _mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng));

        t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
        __m256 valid = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_LT_OS);
        while (_mm256_testz_si256(_mm256_castps_si256(valid), _mm256_castps_si256(valid)) == 0) {
            xi1 = _mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng));
            xi2 = _mm256_mul_ps(_mm256_set1_ps(2.0f), xorshift32_randf8(rng));
            t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
            valid = _mm256_cmp_ps(t, _mm256_set1_ps(1.0f), _CMP_LT_OS);
        }
        __m256 sqrt_factor = _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(xi1, xi1)), t));
        __m256 new_u = _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(xi1, sqrt_factor));
        __m256 new_v = _mm256_mul_ps(xi1, sqrt_factor);
        __m256 new_w = _mm256_mul_ps(xi2, sqrt_factor);
        _mm256_storeu_ps(&p->u[index], new_u);
        _mm256_storeu_ps(&p->v[index], new_v);
        _mm256_storeu_ps(&p->w[index], new_w);

        /* roulette */
        __m256 roulette_weight = _mm256_loadu_ps(&p->weight[index]);
        __m256 is_below_threshold = _mm256_cmp_ps(roulette_weight, _mm256_set1_ps(roulette_threshold), _CMP_LT_OS);
        if (_mm256_testz_si256(_mm256_castps_si256(is_below_threshold), _mm256_castps_si256(is_below_threshold)) == 0) {
            __m256 rand_vals = xorshift32_randf8(rng);
            __m256 survival_check = _mm256_cmp_ps(rand_vals, _mm256_set1_ps(roulette_survival_prob), _CMP_GT_OS);
            roulette_weight = _mm256_blendv_ps(roulette_weight, _mm256_mul_ps(roulette_weight, _mm256_set1_ps(roulette_weight_boost)), survival_check);
            _mm256_storeu_ps(&p->weight[index], roulette_weight);
            __m256 all_dead_check = _mm256_cmp_ps(roulette_weight, _mm256_set1_ps(0.0f), _CMP_EQ_OS);
            if (_mm256_testz_si256(_mm256_castps_si256(all_dead_check), _mm256_castps_si256(all_dead_check)) == 0) {
                break;
            }
        }
    }

    _mm256_storeu_ps(&p->x[index], x);
    _mm256_storeu_ps(&p->y[index], y);
    _mm256_storeu_ps(&p->z[index], z);
    _mm256_storeu_ps(&p->u[index], u);
    _mm256_storeu_ps(&p->v[index], v);
    _mm256_storeu_ps(&p->w[index], w);
    _mm256_storeu_ps(&p->weight[index], weight);
}
