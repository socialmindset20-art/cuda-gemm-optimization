NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -I./include

# Adicionamos o tiled.cu aqui!
SRC_FILES = src/main.cu src/kernels/naive.cu src/kernels/tiled.cu
TARGET = gemm_runner

all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
