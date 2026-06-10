
#include <iostream>
#include "timer.hpp"
#include <cuda_runtime.h>

#include "image_io.hpp"
#include "convolution.cuh"

int main()
{
    int width,height,channels;

    unsigned char* image =
        loadImage(
            "data/input.jpg",
            width,
            height,
            channels);

    size_t bytes =
        width*height*channels;

    unsigned char *d_input,*d_output;

    cudaMalloc(&d_input,bytes);
    cudaMalloc(&d_output,bytes);

    cudaMemcpy(
        d_input,
        image,
        bytes,
        cudaMemcpyHostToDevice);

    launchConvolution(
        d_input,
        d_output,
        width,
        height,
        channels);

    unsigned char* result =
        new unsigned char[bytes];

    cudaMemcpy(
        result,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost);

    saveImage(
        "data/output.jpg",
        result,
        width,
        height,
        channels);

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] result;

    return 0;
}

