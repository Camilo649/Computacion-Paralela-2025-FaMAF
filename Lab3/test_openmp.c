#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Hola desde el hilo %d\n", tid);
    }
    printf("OpenMP version: %d\n", _OPENMP);
    return 0;
}

