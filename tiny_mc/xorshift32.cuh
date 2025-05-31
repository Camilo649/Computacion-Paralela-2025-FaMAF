#ifndef XORSHIFT32_CUH
#define XORSHIFT32_CUH

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
__device__ __inline__ void xorshift32_init(Xorshift32* rng, uint32_t seed) {
    rng->state = seed;
}


/**
* @brief Genera un número aleatorio de punto flotante en el rango [0, 1) utilizando Xorshift32.
* 
* @param rng Puntero al generador Xorshift32 que mantiene el estado.
* @return float Un número aleatorio de punto flotante en el rango [0, 1).
*/
__device__ __inline__ float xorshift32_norm(Xorshift32* rng) {
    rng->state ^= rng->state << 13;
    rng->state ^= rng->state >> 17;
    rng->state ^= rng->state << 5;
    return rng->state * (1.0f / 4294967296.0f);
}

#endif // XORSHIFT32_CUH
