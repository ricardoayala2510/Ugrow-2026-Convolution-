#include <cuda_runtime.h>

__constant__ float d_blurKernel[9] =
{
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
};

__constant__ float d_sharpenKernel[9] =
{
     0.0f, -1.0f,  0.0f,
    -1.0f,  5.0f, -1.0f,
     0.0f, -1.0f,  0.0f
};

__device__ __forceinline__ unsigned char clampToUChar(float value)
{
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<unsigned char>(value + 0.5f);
}

__global__
void convolutionKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int kernelType)       // 0 = blur, 1 = sharpen
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < channels; c++)
    {
        // If the image has an alpha channel, keep alpha unchanged.
        if (channels == 4 && c == 3)
        {
            int idx = (y * width + x) * channels + c;
            output[idx] = input[idx];
            continue;
        }

        float sum = 0.0f;

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);

                int inputIdx = (ny * width + nx) * channels + c;
                int kernelIdx = (ky + 1) * 3 + (kx + 1);

                float coeff =
                    (kernelType == 0)
                    ? d_blurKernel[kernelIdx]
                    : d_sharpenKernel[kernelIdx];

                sum += input[inputIdx] * coeff;
            }
        }

        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = clampToUChar(sum);
    }
}

static void launchConvolutionWithKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY,
    int kernelType)
{
    dim3 block(blockX, blockY);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    convolutionKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height,
        channels,
        kernelType);
}

void launchBlur(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY)
{
    launchConvolutionWithKernel(
        d_input,
        d_output,
        width,
        height,
        channels,
        blockX,
        blockY,
        0);
}

void launchSharpen(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY)
{
    launchConvolutionWithKernel(
        d_input,
        d_output,
        width,
        height,
        channels,
        blockX,
        blockY,
        1);
}
