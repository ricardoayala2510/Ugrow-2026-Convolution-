NVCC = nvcc
TARGET = conv
INCLUDES = -Iinclude
SOURCES = src/main.cpp src/image_io.cpp src/convolution.cu
CFLAGS = -O3 -std=c++17

all:
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f gpu_benchmark_results.csv
	rm -f data/output_blur.jpg data/output_sharpen.jpg data/output_edge_detection.jpg data/output_emboss.jpg
