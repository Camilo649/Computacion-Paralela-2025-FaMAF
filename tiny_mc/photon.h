#include "xorshift32.h"

#pragma once


/**
* @brief Simula la propagación de múltiples fotones en un medio dispersivo y absorbente.
*
* @param heats             Arreglo que acumula la energía absorbida por shell.
* @param heats_squared     Arreglo que acumula el cuadrado de la energía absorbida (para varianza).
* @param rng               Puntero al generador de números aleatorios (Xorshift32).
*/
int photon(float *heats, float *heats_squared, Xorshift32 * restrict rng);
