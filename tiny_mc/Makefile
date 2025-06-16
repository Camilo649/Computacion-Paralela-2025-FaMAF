# Makefile para proyecto CUDA + C con compilador Intel (icx)

# Compiladores
CC      = icx
NVCC    = nvcc

# Arquitectura del CPU Intel Xeon E5-2680 v4 (Broadwell con AVX2 y FMA)
ARCH_FLAGS = -march=broadwell -mtune=broadwell

# Flags para icx (CPU C code)
CFLAGS = -Wall -Wextra -Werror -O3 $(ARCH_FLAGS) \
         -ffast-math -funroll-loops -fimf-use-svml=1 \
         -mavx2 -mfma -falign-functions=32 -fPIC

# Flags para nvcc (CUDA code)
NVCCFLAGS = -O3 -gencode arch=compute_75,code=sm_75 \
            -Xcompiler "$(CFLAGS)" \
            --compiler-bindir=/opt/intel/oneapi/compiler/2025.0/bin/icx \
            -allow-unsupported-compiler

# Objetos
OBJS = tiny_mc.o photon.o xorshift32.o

# Target final
TARGET = mixed_sim

# Regla principal
all: $(TARGET)

# Compilar archivo CUDA
tiny_mc.o: tiny_mc.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compilar módulos C con icx
photon.o: photon.c
	$(CC) $(CFLAGS) -c $< -o $@

xorshift32.o: xorshift32.c
	$(CC) $(CFLAGS) -c $< -o $@

# Linking final con nvcc para mezclar C + CUDA y OpenMP
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "-qopenmp" -o $@ $^ -lm

# Limpieza
clean:
	rm -f *.o $(TARGET)

.PHONY: all clean

