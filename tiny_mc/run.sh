#!/bin/bash

# Valor inicial de PHOTONS
PHOTONS=65536
# Saque la variable i

# Archivo de salida
OUTPUT_FILE="results.csv"

# Escribir encabezado en el archivo CSV
echo "PHOTONS,Kphotons/seg" > $OUTPUT_FILE
# Iterar 10 veces
for _ in {1..10}; do 
    echo "Running with PHOTONS=$PHOTONS"
    
    # Compilar con el nuevo valor de PHOTONS
    make clean > /dev/null
    make CFLAGS="-g -std=c11 -ffast-math  -Wall -Wextra -Werror -O3 -march=native -DPHOTONS=$PHOTONS" > /dev/null
    
    # Ejecutar el programa y guardar la salida en una variable
    KFOTONES=$(taskset -c 0,1 ./headless)
    
    # Escribir en CSV
    echo "$PHOTONS,$KFOTONES" >> $OUTPUT_FILE
    
    # Duplicar el número de PHOTONS
    PHOTONS=$((PHOTONS * 2))
    ((i++))
done

