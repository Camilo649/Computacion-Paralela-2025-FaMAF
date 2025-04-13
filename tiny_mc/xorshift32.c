#include "xorshift32.h"
#include <immintrin.h>
#include <time.h>

// Funcion auxiliar para mezclar las semillas
uint32_t splitmix32(uint32_t x) {
    x += 0x9e3779b9;
    x = (x ^ (x >> 15)) * 0x85ebca6b;
    x = (x ^ (x >> 13)) * 0xc2b2ae35;
    return x ^ (x >> 16);
}


// Inicialización del generador Xorshift32
void xorshift32_init(Xorshift32* rng)
{
    uint32_t seed = time(NULL);
    for (int i = 0; i < 8; ++i) {
        rng->state[i] = splitmix32(seed + i);
    }
}

// Generador de número aleatorio de 32 bits
// El generador tiene periódo maximo según el siguiente articulo: https://en.wikipedia.org/wiki/Xorshift
uint32_t xorshift32_next(Xorshift32* rng, size_t st)
{
    uint32_t s = rng->state[st];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    rng->state[st] = s;
    return s;
}

// Función para generar un número aleatorio de 32 bits en el rango [0, 1)
float xorshift32_randf(Xorshift32* rng)
{
    return (float)xorshift32_next(rng,0) / (float)UINT32_MAX;
}

// Función SIMD para generar 8 números aleatorios de 32 bits
__m256 xorshift32_randf8(Xorshift32* rng)
{
    uint32_t random_values[8];

    // Generamos 8 números aleatorios y los almacenamos en un array
    for (int i = 0; i < 8; ++i) {
        random_values[i] = xorshift32_next(rng,i);
    }

    // Devolvemos los 8 números en un vector de 256 bits (8x32-bit floats)
    // Ordenados de arriba abajo para que se cumpla que la posicion i-ésima del vector sea el estado i-ésimo 
    return  _mm256_set_ps(
        (float)random_values[7] / (float)UINT32_MAX,
        (float)random_values[6] / (float)UINT32_MAX,
        (float)random_values[5] / (float)UINT32_MAX,
        (float)random_values[4] / (float)UINT32_MAX,
        (float)random_values[3] / (float)UINT32_MAX,
        (float)random_values[2] / (float)UINT32_MAX,
        (float)random_values[1] / (float)UINT32_MAX,
        (float)random_values[0] / (float)UINT32_MAX
    );
}