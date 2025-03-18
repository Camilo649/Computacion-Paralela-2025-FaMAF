#!/bin/bash

# Valor inicial de PHOTONS
PHOTONS=65536
i=1

# Archivo de salida
OUTPUT_FILE="results.csv"

# Escribir encabezado en el archivo CSV
echo "PHOTONS,Kphotons/seg" > $OUTPUT_FILE

while [ $i -le 10 ]; do
    echo "Running with PHOTONS=$PHOTONS"
    
    # Compilar con el nuevo valor de PHOTONS
    make clean
    make CFLAGS="-g -std=c11 -Wall -Wextra -Werror -O3 -march=native -DPHOTONS=$PHOTONS"
    
    # Ejecutar el programa y guardar la salida en una variable
    KFOTONES=$(./headless)
    
    # Escribir en CSV
    echo "$PHOTONS,$KFOTONES" >> $OUTPUT_FILE
    
    # Duplicar el número de PHOTONS
    PHOTONS=$((PHOTONS * 2))
    ((i++))
done

