/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
* 1 W Point Source Heating in Infinite Isotropic Scattering Medium
* http://omlc.ogi.edu/software/mc/tiny_mc.c
*
* Adaptado para CP2014, Nicolas Wolovick
*/

#define _XOPEN_SOURCE 500 // M_PI

#include <GL/glew.h>
#include "xorshift32.h"
#include "params.h"
#include "photon.h"
 
#include <assert.h>
// #include <math.h>
#include <stdio.h>
// #include <stdlib.h>
#include <omp.h>
 
// char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
// char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
// char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
//             " and Nicolas Wolovick";
 
// global state, heat and heat square in each shell
float heat[SHELLS] = { 0 };
float heat2[SHELLS] = { 0 };
 
/***
 * Main matter
***/
 
int main(void)
{
    // heading
    // printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    // printf("# Scattering = %8.3f/cm\n", MU_S);
    // printf("# Absorption = %8.3f/cm\n", MU_A);
    // printf("# Photons    = %8d\n#\n", PHOTONS);
 
    // start timer
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        Xorshift32 rng;
        xorshift32_init(&rng);

        float local_heat[SHELLS] = {0};
        float local_heat2[SHELLS] = {0};

	    size_t total_iterations = PHOTONS / (8 * CHUNK_SIZE);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < total_iterations; ++i) {
            photon8(&rng, local_heat, local_heat2, CHUNK_SIZE);
        }

        #pragma omp critical
        {
            for (int j = 0; j < SHELLS; ++j) {
                heat[j] += local_heat[j];
                heat2[j] += local_heat2[j];
            }
        }
    }

    // stop timer
    double end = omp_get_wtime();
    assert(start <= end);
    double elapsed = end - start;
 
    // printf("# %f seconds\n", elapsed);
    printf("%f\n", 1e-6 * PHOTONS / elapsed);
 
    // printf("# Radius\tHeat\n");
    // printf("# [microns]\t[W/cm^3]\tError\n");
    // float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    // for (unsigned int i = 0; i < SHELLS - 1; ++i) {
    //     printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
    //            heat[i] / t / (i * i + i + 1.0 / 3.0),
    //            sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    // }
    // printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
 
    return 0;
}

