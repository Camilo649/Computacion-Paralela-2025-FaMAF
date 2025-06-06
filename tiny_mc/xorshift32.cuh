#ifndef XORSHIFT32_CUH
#define XORSHIFT32_CUH

#include <stdint.h>

#define INV_2PI32 2.3283064365386963e-10f // == 1.0f / 4294967296.0f

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
    uint32_t x = rng->state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    rng->state = x;

    return __uint2float_rn(x) * INV_2PI32;
}

#endif // XORSHIFT32_CUH
