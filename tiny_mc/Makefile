# Compilers
NVCC = nvcc

# Flags
NVFLAGS = -O3 -use_fast_math -lineinfo -Xcompiler "-fno-stack-protector -funroll-loops -ffast-math -fstrict-aliasing" \
          -arch=sm_61 -code=sm_61 -restrict -Xptxas "-O3 -v" -DSEED=666666
TINY_LDFLAGS = -lm
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = headless head

# Files
C_SOURCES = photon.cu xorshift32.cu
C_OBJS = $(patsubst %.cu, %.o, $(C_SOURCES))

# Default rules
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

headless: tiny_mc.o $(C_OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o && rm -rf vtune_results r000hs
