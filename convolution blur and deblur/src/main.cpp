#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#include "image_io.hpp"
#include "convolution.cuh"

#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

struct BlockConfig
{
    int x;
    int y;
};

struct PerfRow
{
    int testNumber;
    std::string imageSize;
    std::string blockSize;
    std::string operation;
    float kernelTimeMs;
    double throughputMPixelsPerSec;
};

std::string makeImageSizeString(int width, int height)
{
    std::ostringstream ss;
    ss << width << "x" << height;
    return ss.str();
}

std::string makeBlockSizeString(int blockX, int blockY)
{
    std::ostringstream ss;
    ss << blockX << "x" << blockY;
    return ss.str();
}

float timeBlurOnly(
    const unsigned char* d_input,
    unsigned char* d_blur,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY,
    int iterations)
{
    // Warm-up launch. This avoids measuring one-time GPU setup overhead.
    launchBlur(d_input, d_blur, width, height, channels, blockX, blockY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        launchBlur(d_input, d_blur, width, height, channels, blockX, blockY);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / iterations;
}

float timeBlurThenSharpen(
    const unsigned char* d_input,
    unsigned char* d_blur,
    unsigned char* d_enhanced,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY,
    int iterations)
{
    // Warm-up launch. The sharpen stage takes the blurred image as input.
    launchBlur(d_input, d_blur, width, height, channels, blockX, blockY);
    launchSharpen(d_blur, d_enhanced, width, height, channels, blockX, blockY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        // Correct sequential pipeline:
        // original GPU image -> blurred GPU image -> enhanced GPU image
        launchBlur(d_input, d_blur, width, height, channels, blockX, blockY);
        launchSharpen(d_blur, d_enhanced, width, height, channels, blockX, blockY);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / iterations;
}

void writePerformanceTable(std::ostream& out, const std::vector<PerfRow>& rows)
{
    out << std::left
        << std::setw(8)  << "Test"
        << std::setw(14) << "Image Size"
        << std::setw(14) << "Block Size"
        << std::setw(20) << "Operation"
        << std::right
        << std::setw(18) << "Kernel Time ms"
        << std::setw(26) << "Throughput MPixels/s"
        << "\n";

    out << std::string(100, '-') << "\n";

    for (const PerfRow& row : rows)
    {
        out << std::left
            << std::setw(8)  << row.testNumber
            << std::setw(14) << row.imageSize
            << std::setw(14) << row.blockSize
            << std::setw(20) << row.operation
            << std::right
            << std::setw(18) << std::fixed << std::setprecision(4) << row.kernelTimeMs
            << std::setw(26) << std::fixed << std::setprecision(2) << row.throughputMPixelsPerSec
            << "\n";
    }
}

int main()
{
    int width, height, channels;

    unsigned char* image = loadImage(
        "data/input.jpg",
        width,
        height,
        channels);

    if (image == nullptr)
    {
        std::cerr << "Error: could not load data/input.jpg" << std::endl;
        return 1;
    }

    size_t bytes = static_cast<size_t>(width) * height * channels;

    unsigned char *d_input, *d_blur, *d_enhanced;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_blur, bytes));
    CUDA_CHECK(cudaMalloc(&d_enhanced, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        image,
        bytes,
        cudaMemcpyHostToDevice));

    const int iterations = 100;
    const std::string imageSize = makeImageSizeString(width, height);

    std::vector<BlockConfig> blockConfigs =
    {
        {8, 8},
        {16, 16},
        {32, 8},
        {32, 16}
    };

    std::vector<PerfRow> rows;
    int testNumber = 1;

    for (const BlockConfig& block : blockConfigs)
    {
        float blurMs = timeBlurOnly(
            d_input,
            d_blur,
            width,
            height,
            channels,
            block.x,
            block.y,
            iterations);

        double blurThroughput =
            (static_cast<double>(width) * height / 1.0e6) /
            (blurMs / 1000.0);

        rows.push_back({
            testNumber++,
            imageSize,
            makeBlockSizeString(block.x, block.y),
            "Blur only",
            blurMs,
            blurThroughput
        });

        float blurSharpenMs = timeBlurThenSharpen(
            d_input,
            d_blur,
            d_enhanced,
            width,
            height,
            channels,
            block.x,
            block.y,
            iterations);

        // Blur + sharpen is two full image-processing passes.
        double blurSharpenThroughput =
            (2.0 * static_cast<double>(width) * height / 1.0e6) /
            (blurSharpenMs / 1000.0);

        rows.push_back({
            testNumber++,
            imageSize,
            makeBlockSizeString(block.x, block.y),
            "Blur + sharpen",
            blurSharpenMs,
            blurSharpenThroughput
        });
    }

    std::ofstream report("data/performance_results.txt");
    if (!report)
    {
        std::cerr << "Error: could not create data/performance_results.txt" << std::endl;
        return 1;
    }

    report << "CUDA Convolution Performance Analysis\n";
    report << "Input image: data/input.jpg\n";
    report << "Output images: data/output_blur.jpg and data/output_enhanced.jpg\n";
    report << "Image size: " << imageSize << "\n";
    report << "Channels: " << channels << "\n";
    report << "Timing method: CUDA events\n";
    report << "Iterations per test: " << iterations << "\n";
    report << "Note: kernel time excludes image loading, image saving, cudaMalloc, and cudaMemcpy.\n";
    report << "Note: enhanced image is computed sequentially as sharpen(blur(original)).\n\n";

    writePerformanceTable(report, rows);
    report.close();

    std::cout << "\nCUDA Convolution Performance Analysis\n";
    std::cout << "Results written to data/performance_results.txt\n\n";
    writePerformanceTable(std::cout, rows);

    // Save final images using 16x16 as the standard CUDA block size.
    launchBlur(d_input, d_blur, width, height, channels, 16, 16);
    launchSharpen(d_blur, d_enhanced, width, height, channels, 16, 16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* blurResult = new unsigned char[bytes];
    unsigned char* enhancedResult = new unsigned char[bytes];

    CUDA_CHECK(cudaMemcpy(
        blurResult,
        d_blur,
        bytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        enhancedResult,
        d_enhanced,
        bytes,
        cudaMemcpyDeviceToHost));

    saveImage(
        "data/output_blur.jpg",
        blurResult,
        width,
        height,
        channels);

    saveImage(
        "data/output_enhanced.jpg",
        enhancedResult,
        width,
        height,
        channels);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_enhanced));

    delete[] blurResult;
    delete[] enhancedResult;

    return 0;
}
