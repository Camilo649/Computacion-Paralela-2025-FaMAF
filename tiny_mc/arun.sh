#!/bin/bash
#SBATCH --job-name=photon_sim       # Nombre del trabajo
#SBATCH --output=slurm_output_%j.log # Archivo de salida (%j = Job ID)
#SBATCH --error=slurm_error_%j.log   # Archivo de error
#SBATCH --time=00:10:00              # Tiempo máximo de ejecución (HH:MM:SS)
#SBATCH --ntasks=1                   # Número de tareas (procesos)
#SBATCH --cpus-per-task=1            # Número de CPUs asignadas
#SBATCH --mem=1G                     # Memoria solicitada
#SBATCH --partition=default          # Cola/partición del clúster (ajústalo según el clúster)

# Valor inicial de PHOTONS
PHOTONS=65536

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

