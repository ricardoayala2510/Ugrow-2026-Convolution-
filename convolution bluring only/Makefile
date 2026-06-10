
NVCC = nvcc

TARGET = conv

INCLUDES = -Iinclude

SOURCES = \
src/main.cpp \
src/image_io.cpp \
src/convolution.cu

CFLAGS = -O3

all:
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)




