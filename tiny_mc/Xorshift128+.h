#include <stdint.h>

#pragma once

// Estructura para almacenar el estado del generador Xorshift128+
typedef struct Xorshift128Plus {
    uint64_t state[2];  // Estado interno del generador (dos valores de 64 bits)
} Xorshift128Plus;

/**
 * @brief Inicializa el generador Xorshift128+ con dos semillas de 64 bits.
 * 
 * @param rng Puntero al generador Xorshift128+ a inicializar.
 * @param seed1 Primer valor de semilla (64 bits).
 * @param seed2 Segundo valor de semilla (64 bits).
 * 
 * @note El generador usará estos dos valores para inicializar su estado interno.
 */
void xorshift128plus_init(Xorshift128Plus* rng, uint64_t seed1, uint64_t seed2);

/**
 * @brief Genera el siguiente número aleatorio de 64 bits utilizando Xorshift128+.
 * 
 * @param rng Puntero al generador Xorshift128+ que mantiene el estado.
 * @return uint64_t El siguiente número aleatorio de 64 bits.
 * 
 * @note Esta función ctualiza el estado del generador.
 */
uint64_t xorshift128plus_next(Xorshift128Plus* rng);

/**
 * @brief Genera un número aleatorio de punto flotante en el rango [0, 1) utilizando Xorshift128+.
 * 
 * @param rng Puntero al generador Xorshift128+ que mantiene el estado.
 * @return float Un número aleatorio de punto flotante en el rango [0, 1).
 * 
 * @note Esta función convierte el número de 64 bits generado por `xorshift128plus_next` a un número flotante de simple precision en el rango [0, 1).
 */
float xorshift128plus_randf(Xorshift128Plus* rng);
