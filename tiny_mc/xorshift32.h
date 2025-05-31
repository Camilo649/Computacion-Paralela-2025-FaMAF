#ifndef XORSHIFT32_H
#define XORSHIFT32_H

#include <stdint.h>

/**
* @brief Estructura para almacenar el estado del generador Xorshift32.
*/
typedef struct {
    uint32_t state;
} Xorshift32;


/**
* @brief Inicializa el generador Xorshift32 con una semilla de 32 bits.
* 
* @param rng Puntero al generador Xorshift32 a inicializar.
* @param seed Valor de semilla (32 bits).
* 
* @note seed debe ser un valor != 0
*/
void xorshift32_init(Xorshift32 * restrict rng, uint32_t seed);

/**
* @brief Genera el siguiente número aleatorio de 32 bits utilizando Xorshift32.
* 
* @param rng Puntero al generador Xorshift32 que mantiene el estado..
* @return uint32_t El siguiente número aleatorio de 32 bits.
*/
uint32_t xorshift32(Xorshift32 * restrict rng);

/**
* @brief Genera un número aleatorio de punto flotante en el rango [0, 1) utilizando Xorshift32.
* 
* @param rng Puntero al generador Xorshift32 que mantiene el estado.
* @return float Un número aleatorio de punto flotante en el rango [0, 1).
*/
float xorshift32_norm(Xorshift32 * restrict rng);

#endif // XORSHIFT32_H
