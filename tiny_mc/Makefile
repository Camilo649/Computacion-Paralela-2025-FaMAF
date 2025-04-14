# Compilers
CC = icx

# Flags
CFLAGS = -g -Wall -Wextra -Werror -O3 -march=znver3 -mtune=znver3 -ffast-math -funroll-loops -flto -fvectorize -funsafe-math-optimizations -DSEED=1234567
TINY_LDFLAGS = -lm
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TINY_LDFLAGS = -lm
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = headless head

# Files
C_SOURCES = wtime.c photon.c xorshift32.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

# VTune analysis
VTUNE_RESULTS_DIR = vtune_results

headless: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

# VTune analysis

## Detecta los cuellos de botella en el código y analiza el uso del CPU.
vtune-hotspots:
	rm -rf $(VTUNE_RESULTS_DIR) r000hs
	vtune -collect hotspots -result-dir $(VTUNE_RESULTS_DIR) ./headless
	vtune -report summary -format=csv -r $(VTUNE_RESULTS_DIR) > hotspots.csv

## Analiza el rendimiento relacionado con el acceso a memoria, incluyendo la latencia de caché y problemas
## de acceso a la memoria.
vtune-memory-access:
	rm -rf $(VTUNE_RESULTS_DIR) r000hs
	vtune -collect memory-access -result-dir $(VTUNE_RESULTS_DIR) ./headless
	vtune -report summary -format=csv -r $(VTUNE_RESULTS_DIR) > memory-access.csv

clean:
	rm -f $(TARGETS) *.o && rm -rf vtune_results r000hs
