#include "Xorshift128+.h"
#include "params.h"

#pragma once

typedef struct {
    float x, y, z;   // Posición
    float u, v, w;   // Dirección
    float weight;    // Peso del fotón
} Photon;

// array of photons

extern Photon photons[];

void photon(unsigned int index, Xorshift128Plus* rng, float *heats, float *heats_squared);
