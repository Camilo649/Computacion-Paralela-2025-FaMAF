#ifndef XORSHIFT32_H
#define XORSHIFT32_H

#include <stdint.h>
#include <immintrin.h>

/**
 * @brief Estructura para almacenar el estado del generador Xorshift32.
 */
typedef struct Xorshift32 {
    uint32_t state[8] __attribute__((aligned(32)));  // Alineación de la matriz de estados
} Xorshift32;

/**
 * @brief Inicializa el generador Xorshift32 con una semilla de 32 bits.
 * 
 * @param rng Puntero al generador Xorshift32 a inicializar.
 * @param seed Valor de semilla (32 bits).
 */
void xorshift32_init(Xorshift32* rng);

/**
 * @brief Genera el siguiente número aleatorio de 32 bits utilizando Xorshift32.
 * 
 * @param rng Puntero al generador Xorshift32 que mantiene el estado.
 * @param st Inidice del estado a utilizar.
 * @return uint32_t El siguiente número aleatorio de 32 bits.
 * 
 * @note Esta función actualiza el estado del generador cuyo indice debe ser un valor entre 0-7.
 */
uint32_t xorshift32_next(Xorshift32* rng, size_t st);

/**
 * @brief Genera un número aleatorio de punto flotante en el rango [0, 1) utilizando Xorshift32.
 * 
 * @param rng Puntero al generador Xorshift32 que mantiene el estado.
 * @return float Un número aleatorio de punto flotante en el rango [0, 1).
 * 
 * @note Por defecto, utiliza el primer estado del generador
 */
float xorshift32_randf(Xorshift32* rng);

/**
 * @brief Genera 8 números aleatorios de 32 bits en paralelo utilizando Xorshift32 y SIMD (AVX).
 * 
 * @param rng Puntero al generador Xorshift32 que mantiene el estado.
 * @return __m256 Un vector de 256 bits que contiene 8 números flotantes aleatorios de 32 bits en el rango [0, 1).
 */
__m256 xorshift32_randf8(Xorshift32* rng);

#endif // XORSHIFT32_H
