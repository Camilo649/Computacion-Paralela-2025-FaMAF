#!/bin/bash

# Ejecutar cada binario en segundo plano y guardar sus PIDs
./head_original & PID1=$!
./head_lab1 & PID2=$!
./head_lab2 & PID3=$!

sleep 2

# Buscar ventanas que coincidan con el nombre de la simulación
wm_output=$(wmctrl -lp | grep "tiny mc")

# Extraer ID de ventana asociado a cada PID
get_window_id() {
    local pid=$1
    echo "$wm_output" | awk -v pid="$pid" '$3 == pid { print $1; exit }'
}

WID1=$(get_window_id $PID1)
WID2=$(get_window_id $PID2)
WID3=$(get_window_id $PID3)

# Definir las posiciones en pantalla
X=(0 935 1735)
Y=(0 0 0)

# Mover cada ventana a su lugar
wmctrl -ir $WID1 -e 0,${X[0]},${Y[0]},800,800
wmctrl -ir $WID2 -e 0,${X[1]},${Y[1]},800,800
wmctrl -ir $WID3 -e 0,${X[2]},${Y[2]},800,800

