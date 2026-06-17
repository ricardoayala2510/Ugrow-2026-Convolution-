#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "timer.hpp"
#include "image_io.hpp"
#include "convolution.cuh"

// GPU-only benchmark settings
static constexpr int MIN_POWER = 1;   // 2^1 = 2 images
static constexpr int MAX_POWER = 10;  // 2^10 = 1024 images
static constexpr int NUM_FILTERS = 4;

void checkCuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        std::exit(1);
    }
}

const char* filterName(FilterType filter)
{
    switch (filter)
    {
        case FILTER_BLUR: return "Blur";
        case FILTER_SHARPEN: return "Sharpen";
        case FILTER_EDGE_DETECTION: return "Edge Detection";
        case FILTER_EMBOSS: return "Emboss";
        default: return "Unknown";
    }
}

const char* outputFilename(FilterType filter)
{
    switch (filter)
    {
        case FILTER_BLUR: return "data/output_blur.jpg";
        case FILTER_SHARPEN: return "data/output_sharpen.jpg";
        case FILTER_EDGE_DETECTION: return "data/output_edge_detection.jpg";
        case FILTER_EMBOSS: return "data/output_emboss.jpg";
        default: return "data/output.jpg";
    }
}

// Runs the four CUDA convolutions for imageCount repeated uses of the same image.
// It does not save 1024 outputs. It only writes to four GPU output buffers.
double benchmarkGPUOnly(
    unsigned char* d_input,
    unsigned char* d_outputs[NUM_FILTERS],
    int width,
    int height,
    int channels,
    int imageCount,
    const FilterType filters[NUM_FILTERS],
    float& gpuDeviceMs)
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate start failed");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stop failed");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before benchmark failed");

    Timer wallTimer;
    wallTimer.start();

    checkCuda(cudaEventRecord(startEvent), "cudaEventRecord start failed");

    for (int img = 0; img < imageCount; img++)
    {
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            launchConvolution(
                d_input,
                d_outputs[i],
                width,
                height,
                channels,
                filters[i]);

            checkCuda(cudaGetLastError(), "Kernel launch failed");
        }
    }

    checkCuda(cudaEventRecord(stopEvent), "cudaEventRecord stop failed");
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stop failed");

    wallTimer.stop();

    checkCuda(cudaEventElapsedTime(&gpuDeviceMs, startEvent, stopEvent), "cudaEventElapsedTime failed");
    checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy start failed");
    checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stop failed");

    return wallTimer.milliseconds();
}

void runOneSetAndSaveFourOutputs(
    unsigned char* d_input,
    unsigned char* d_outputs[NUM_FILTERS],
    std::vector<unsigned char>& hostOutput,
    int width,
    int height,
    int channels,
    size_t bytes,
    const FilterType filters[NUM_FILTERS])
{
    std::cout << "\nSaving only 4 sample outputs, not 1024 outputs..." << std::endl;

    for (int i = 0; i < NUM_FILTERS; i++)
    {
        launchConvolution(d_input, d_outputs[i], width, height, channels, filters[i]);
        checkCuda(cudaGetLastError(), "Sample output kernel launch failed");
        checkCuda(cudaDeviceSynchronize(), "Sample output kernel execution failed");

        checkCuda(
            cudaMemcpy(hostOutput.data(), d_outputs[i], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy sample output to CPU failed");

        saveImage(outputFilename(filters[i]), hostOutput.data(), width, height, channels);

        std::cout << "Saved " << outputFilename(filters[i]) << std::endl;
    }
}

int main()
{
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char* image = loadImage("data/input.jpg", width, height, channels);

    if (image == nullptr)
    {
        std::cerr << "Failed to load data/input.jpg" << std::endl;
        return 1;
    }

    const size_t bytes =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        static_cast<size_t>(channels);

    std::cout << "Image size: " << width << " x " << height
              << ", channels: " << channels << std::endl;
    std::cout << "GPU-only benchmark" << std::endl;
    std::cout << "Each image runs 4 CUDA convolutions: Blur, Sharpen, Edge Detection, Emboss" << std::endl;
    std::cout << "Batch sizes: 2, 4, 8, ..., 1024 repeated uses of the same input image" << std::endl;
    std::cout << "Only 4 sample output files are saved. Benchmark does not create 1024 output files.\n" << std::endl;

    const FilterType filters[NUM_FILTERS] =
    {
        FILTER_BLUR,
        FILTER_SHARPEN,
        FILTER_EDGE_DETECTION,
        FILTER_EMBOSS
    };

    unsigned char* d_input = nullptr;
    unsigned char* d_outputs[NUM_FILTERS] = {nullptr, nullptr, nullptr, nullptr};

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes), "cudaMalloc d_input failed");

    for (int i = 0; i < NUM_FILTERS; i++)
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_outputs[i]), bytes), "cudaMalloc d_output failed");
    }

    checkCuda(
        cudaMemcpy(d_input, image, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy input to GPU failed");

    std::vector<unsigned char> hostOutput(bytes);

    // Warm-up so the first measurement is not affected by CUDA initialization.
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        launchConvolution(d_input, d_outputs[i], width, height, channels, filters[i]);
        checkCuda(cudaGetLastError(), "Warm-up kernel launch failed");
    }
    checkCuda(cudaDeviceSynchronize(), "Warm-up execution failed");

    runOneSetAndSaveFourOutputs(
        d_input,
        d_outputs,
        hostOutput,
        width,
        height,
        channels,
        bytes,
        filters);

    std::ofstream csv("gpu_benchmark_results.csv");
    csv << "images,total_convolutions,gpu_wall_ms,gpu_device_ms,ms_per_image,ms_per_convolution,images_per_second,convolutions_per_second\n";

    std::cout << "\nGPU timing results" << std::endl;
    std::cout << std::left
              << std::setw(10) << "Images"
              << std::setw(18) << "Convolutions"
              << std::setw(16) << "GPU wall ms"
              << std::setw(17) << "GPU device ms"
              << std::setw(16) << "ms/image"
              << std::setw(18) << "ms/convolution"
              << std::endl;

    for (int power = MIN_POWER; power <= MAX_POWER; power++)
    {
        const int imageCount = 1 << power;
        const int totalConvolutions = imageCount * NUM_FILTERS;

        float gpuDeviceMs = 0.0f;
        const double gpuWallMs = benchmarkGPUOnly(
            d_input,
            d_outputs,
            width,
            height,
            channels,
            imageCount,
            filters,
            gpuDeviceMs);

        const double msPerImage = gpuWallMs / static_cast<double>(imageCount);
        const double msPerConvolution = gpuWallMs / static_cast<double>(totalConvolutions);
        const double imagesPerSecond = 1000.0 * static_cast<double>(imageCount) / gpuWallMs;
        const double convolutionsPerSecond = 1000.0 * static_cast<double>(totalConvolutions) / gpuWallMs;

        std::cout << std::left
                  << std::setw(10) << imageCount
                  << std::setw(18) << totalConvolutions
                  << std::setw(16) << std::fixed << std::setprecision(3) << gpuWallMs
                  << std::setw(17) << std::fixed << std::setprecision(3) << gpuDeviceMs
                  << std::setw(16) << std::fixed << std::setprecision(6) << msPerImage
                  << std::setw(18) << std::fixed << std::setprecision(6) << msPerConvolution
                  << std::endl;

        csv << imageCount << ","
            << totalConvolutions << ","
            << std::fixed << std::setprecision(6) << gpuWallMs << ","
            << std::fixed << std::setprecision(6) << gpuDeviceMs << ","
            << std::fixed << std::setprecision(9) << msPerImage << ","
            << std::fixed << std::setprecision(9) << msPerConvolution << ","
            << std::fixed << std::setprecision(6) << imagesPerSecond << ","
            << std::fixed << std::setprecision(6) << convolutionsPerSecond << "\n";
    }

    csv.close();

    std::cout << "\nSaved GPU benchmark table to gpu_benchmark_results.csv" << std::endl;
    std::cout << "Use the GPU wall ms column for your comparison chart." << std::endl;

    for (int i = 0; i < NUM_FILTERS; i++)
    {
        cudaFree(d_outputs[i]);
    }

    cudaFree(d_input);
    freeImage(image);

    return 0;
}
