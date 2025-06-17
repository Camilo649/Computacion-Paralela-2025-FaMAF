# Compilers
NVCC = nvcc

# Flags
NVFLAGS = -O3 -use_fast_math -lineinfo -Xcompiler "-fno-stack-protector -funroll-loops -ffast-math -fstrict-aliasing" \
          -gencode arch=compute_75,code=sm_75 -restrict -Xptxas "-O3 -v" -DSEED=666666
TINY_LDFLAGS = -lm
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = headless head

# Sources
TINY_SOURCES = tiny_mc.cu
CG_SOURCES = cg_mc.cu

# Object files
TINY_OBJS = $(patsubst %.cu, %.o, $(TINY_SOURCES))
CG_OBJS = $(patsubst %.cu, %.o, $(CG_SOURCES))

# Default rules
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

headless: $(TINY_OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: $(CG_OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
