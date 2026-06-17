#include <cuda_runtime.h>
#include "convolution.cuh"

#define BLOCK_SIZE 16

// Normal convolution filters: sharpen, edge detection, emboss
#define KERNEL_SIZE 5
#define RADIUS 2

// Strong blur size.
// BLUR_RADIUS 12 = 25x25 blur
// BLUR_RADIUS 20 = 41x41 blur
// BLUR_RADIUS 30 = 61x61 blur
#define BLUR_RADIUS 30

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float d_offset;

__device__
unsigned char clampToByte(float value)
{
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<unsigned char>(value);
}

__device__
int clampInt(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

// Strong blur kernel.
// This averages a large neighborhood around each pixel.
__global__
void strongBlurKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < channels; c++)
    {
        int outputIdx = (y * width + x) * channels + c;

        // Preserve alpha channel if the image has one.
        if (channels == 4 && c == 3)
        {
            output[outputIdx] = input[outputIdx];
            continue;
        }

        float sum = 0.0f;
        int count = 0;

        for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++)
        {
            for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++)
            {
                int nx = clampInt(x + kx, 0, width - 1);
                int ny = clampInt(y + ky, 0, height - 1);

                int inputIdx = (ny * width + nx) * channels + c;

                sum += input[inputIdx];
                count++;
            }
        }

        output[outputIdx] = static_cast<unsigned char>(sum / count);
    }
}

// Normal 5x5 convolution kernel.
// Used for sharpen, edge detection, and emboss.
__global__
void convolutionKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < channels; c++)
    {
        int outputIdx = (y * width + x) * channels + c;

        // Preserve alpha channel if the image has one.
        if (channels == 4 && c == 3)
        {
            output[outputIdx] = input[outputIdx];
            continue;
        }

        float sum = 0.0f;

        for (int ky = -RADIUS; ky <= RADIUS; ky++)
        {
            for (int kx = -RADIUS; kx <= RADIUS; kx++)
            {
                int nx = clampInt(x + kx, 0, width - 1);
                int ny = clampInt(y + ky, 0, height - 1);

                int inputIdx = (ny * width + nx) * channels + c;
                int kernelIdx = (ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS);

                sum += input[inputIdx] * d_kernel[kernelIdx];
            }
        }

        output[outputIdx] = clampToByte(sum + d_offset);
    }
}

void copyFilterToConstantMemory(FilterType filter)
{
 const float sharpen[KERNEL_SIZE * KERNEL_SIZE] =
{
     0.0f,  0.0f, 0.0f,  0.0f,  0.0f,
     0.0f, 0.0f, -1.0f, 0.0f,  0.0f,
    0.0f, -1.0f, 5.0f, -1.0f, 0.0f,
     0.0f, 0.0f, -1.0f, 0.0f,  0.0f,
     0.0f,  0.0f, 0.0f,  0.0f,  0.0f
};


const float edgeDetection[KERNEL_SIZE * KERNEL_SIZE] =
{
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  0.0f,  0.0f,  0.0f, -1.0f,
    -1.0f,  0.0f, 16.0f,  0.0f, -1.0f,
    -1.0f,  0.0f,  0.0f,  0.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
};

 const float emboss[KERNEL_SIZE * KERNEL_SIZE] =
{
    -2.0f, -1.0f,  0.0f,  0.0f,  0.0f,
    -1.0f, -1.0f,  0.0f,  0.0f,  0.0f,
     0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
     0.0f,  0.0f,  0.0f,  1.0f,  1.0f,
     0.0f,  0.0f,  0.0f,  1.0f,  2.0f
};

    const float* selectedKernel = sharpen;
    float offset = 0.0f;

    switch (filter)
    {
        case FILTER_SHARPEN:
            selectedKernel = sharpen;
            offset = 0.0f;
            break;

        case FILTER_EDGE_DETECTION:
            selectedKernel = edgeDetection;
            offset = 0.0f;
            break;

        case FILTER_EMBOSS:
            selectedKernel = emboss;
            offset = 128.0f;
            break;

        case FILTER_BLUR:
        default:
            selectedKernel = sharpen;
            offset = 0.0f;
            break;
    }

    cudaMemcpyToSymbol(
        d_kernel,
        selectedKernel,
        sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);

    cudaMemcpyToSymbol(
        d_offset,
        &offset,
        sizeof(float));
}

void launchConvolution(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    FilterType filter)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    // special large-kernel blur for a strong blur effect.
    if (filter == FILTER_BLUR)
    {
        strongBlurKernel<<<grid, block>>>(
            d_input,
            d_output,
            width,
            height,
            channels);

        return;
    }

    // normal 3x3 convolution for the other filters.
    copyFilterToConstantMemory(filter);

    convolutionKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height,
        channels);
}