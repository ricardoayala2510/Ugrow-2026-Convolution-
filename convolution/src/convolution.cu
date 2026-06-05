#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__constant__ float d_kernel[9] =
{
    1.0f/9,1.0f/9,1.0f/9,
    1.0f/9,1.0f/9,1.0f/9,
    1.0f/9,1.0f/9,1.0f/9
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

    if(x >= width || y >= height)
        return;

    for(int c=0;c<channels;c++)
    {
        float sum = 0.0f;

        for(int ky=-1; ky<=1; ky++)
        {
            for(int kx=-1; kx<=1; kx++)
            {
                int nx = min(max(x+kx,0),width-1);
                int ny = min(max(y+ky,0),height-1);

                int idx =
                    (ny*width+nx)*channels+c;

                sum +=
                    input[idx] *
                    d_kernel[(ky+1)*3+(kx+1)];
            }
        }

        int outIdx =
            (y*width+x)*channels+c;

        output[outIdx] =
            (unsigned char)sum;
    }
}

void launchConvolution(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels)
{
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);

    dim3 grid(
        (width+block.x-1)/block.x,
        (height+block.y-1)/block.y);

    convolutionKernel<<<grid,block>>>(
        d_input,
        d_output,
        width,
        height,
        channels);
}

