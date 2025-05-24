#include "xorshift32.h"
#include "params.h"

#pragma once

/**
 * @brief Simula la propagación de múltiples fotones en un medio dispersivo y absorbente.
 *
 * @param rng               Puntero al generador de números aleatorios (Xorshift32).
 * @param heats             Arreglo que acumula la energía absorbida por shell.
 * @param heats_squared     Arreglo que acumula el cuadrado de la energía absorbida (para varianza).
 * @param photons_loop      Cantidad de fotones procesados.
 */
void photon8(Xorshift32* restrict rng,
    float* restrict heats,
    float* restrict heats_squared,
    size_t photons_loop);

