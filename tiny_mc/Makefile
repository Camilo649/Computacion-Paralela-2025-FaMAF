NVCC = nvcc

NVFLAGS = -O3 -gencode arch=compute_75,code=sm_75 -Xcompiler "-fopenmp -O3 -march=znver3 -ffast-math -funroll-loops"

SRC = tiny_mc.cu photon.cuh xorshift32.cuh photon.c xorshift32.c
OBJ = $(SRC:.c=.o)
OBJ := $(OBJ:.cu=.o)

TARGET = mixed_sim

all: $(TARGET)

%.o: %.cpp
	$(NVCC) $(NVFLAGS) -x c++ -c $< -o $@

%.o: %.c
	$(NVCC) $(NVFLAGS) -x c -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVFLAGS) -o $@ $^ -lm

clean:
	rm -f *.o $(TARGET)
