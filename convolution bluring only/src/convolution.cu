#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 5
#define RADIUS 2

__constant__ float d_kernel[25] =
{
    1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25,
    1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25,
    1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25,
    1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25,
    1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25
};

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
        float sum = 0.0f;

        for (int ky = -RADIUS; ky <= RADIUS; ky++)
        {
            for (int kx = -RADIUS; kx <= RADIUS; kx++)
            {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);

                int inputIdx = (ny * width + nx) * channels + c;
                int kernelIdx = (ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS);

                sum += input[inputIdx] * d_kernel[kernelIdx];
            }
        }

        int outputIdx = (y * width + x) * channels + c;

        output[outputIdx] = (unsigned char)sum;
    }
}

void launchConvolution(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    convolutionKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height,
        channels);
}