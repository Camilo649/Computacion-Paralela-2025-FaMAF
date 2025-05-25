#!/bin/bash

# Valor fijo de PHOTONS
PHOTONS=$((1 << 34))  # 2^34 = 16G

# Valor inicial de CHUNK_SIZE
CHUNK_SIZE=1

# Archivo de salida
OUTPUT_FILE="results.csv"

# Escribir encabezado
echo "CHUNK_SIZE,Mphotons/seg" > "$OUTPUT_FILE"

# Iterar hasta CHUNK_SIZE = 2^19
while [ "$CHUNK_SIZE" -le $((1 << 19)) ]; do
    echo "Running with CHUNK_SIZE=$CHUNK_SIZE"

    # Limpiar y compilar con CHUNK_SIZE y PHOTONS
    make clean > /dev/null
    make CFLAGS="-Wall -Wextra -Werror -O3 -march=znver3 -mtune=znver3 -ffast-math -funroll-loops -flto -fvectorize -funsafe-math-optimizations -fimf-use-svml=1 -fno-stack-protector -mavx2 -falign-functions=32 -fstrict-aliasing -qopenmp -DPHOTONS=$PHOTONS -DCHUNK_SIZE=$CHUNK_SIZE" > /dev/null

    # Ejecutar y capturar la salida
    MPHOTONS=$(./headless)

    # Guardar resultados en CSV
    echo "$CHUNK_SIZE,$MPHOTONS" >> "$OUTPUT_FILE"

    # Duplicar el chunk size
    CHUNK_SIZE=$((CHUNK_SIZE * 2))
done
