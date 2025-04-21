#!/bin/bash

# Valor inicial de PHOTONS
PHOTONS=262144

# Archivo de salida
OUTPUT_FILE="results.csv"

# Escribir encabezado en el archivo CSV
echo "PHOTONS,Kphotons/seg" > $OUTPUT_FILE
# Iterar 10 veces
for _ in {1..10}; do 
    echo "Running with PHOTONS=$PHOTONS"
    
    # Compilar con el nuevo valor de PHOTONS
    make clean > /dev/null
    make CFLAGS="-g -Wall -Wextra -Werror -O3 -march=znver3 -mtune=znver3 -ffast-math -funroll-loops -flto -DPHOTONS=$PHOTONS" > /dev/null
    
    # Ejecutar el programa y guardar la salida en una variable
    KFOTONES=$(./headless)
    
    # Escribir en CSV
    echo "$PHOTONS,$KFOTONES" >> $OUTPUT_FILE
    
    # Duplicar el número de PHOTONS
    PHOTONS=$((PHOTONS * 2))
    ((i++))
done

