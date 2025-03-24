#include <math.h>

#include "Xorshift128+.h"
#include "params.h"
#include "photon.h"

Photon photons[PHOTONS];

void photon(unsigned int index, Xorshift128Plus* rng, float* heats, float* heats_squared)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    Photon p = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f
    };

    photons[index] = p;

    for (;;) {
        float t = -logf(xorshift128plus_randf(rng)); /* move */
        photons[index].x += t * photons[index].u;
        photons[index].y += t * photons[index].v;
        photons[index].z += t * photons[index].w;

        unsigned int shell = sqrtf(photons[index].x * photons[index].x + photons[index].y * photons[index].y + photons[index].z * photons[index].z) * shells_per_mfp; /* absorb */

        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        heats[shell] += (1.0f - albedo) * photons[index].weight;
        heats_squared[shell] += (1.0f - albedo) * (1.0f - albedo) * photons[index].weight * photons[index].weight; /* add up squares */
        photons[index].weight *= albedo;

        /* New direction, rejection method */
        float xi1, xi2;
        do {
            xi1 = 2.0f * xorshift128plus_randf(rng) - 1.0f;
            xi2 = 2.0f * xorshift128plus_randf(rng) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        photons[index].u = 2.0f * t - 1.0f;
        photons[index].v = xi1 * sqrtf((1.0f - photons[index].u * photons[index].u) / t);
        photons[index].w = xi2 * sqrtf((1.0f - photons[index].u * photons[index].u) / t);

        if (photons[index].weight < 0.001f) { /* roulette */
            if (xorshift128plus_randf(rng) > 0.1f)
                break;
            photons[index].weight /= 0.1f;
        }
    }
}
