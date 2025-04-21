#include "xorshift32.h"
#include "params.h"

#pragma once

typedef struct {
    float x[PHOTONS];
    float y[PHOTONS];
    float z[PHOTONS];
    float u[PHOTONS];
    float v[PHOTONS];
    float w[PHOTONS];
    float weight[PHOTONS];
} Photons;

/**
 * @brief Simula la propagación de múltiples fotones en un medio dispersivo y absorbente.
 *
 * @param rng               Puntero al generador de números aleatorios (Xorshift32).
 * @param p                 Puntero a la estructura que almacenará el estado final de los fotones.
 * @param heats             Arreglo que acumula la energía absorbida por shell.
 * @param heats_squared     Arreglo que acumula el cuadrado de la energía absorbida (para varianza).
 * @param index             Índice del foton.
 */
void photon8(Xorshift32* restrict rng,
    Photons* restrict p,
    float* restrict heats,
    float* restrict heats_squared,
    size_t index);

