#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

unsigned char* loadImage(
    const char* filename,
    int& width,
    int& height,
    int& channels)
{
    return stbi_load(
        filename,
        &width,
        &height,
        &channels,
        0);
}

void saveImage(
    const char* filename,
    unsigned char* data,
    int width,
    int height,
    int channels)
{
    stbi_write_jpg(
        filename,
        width,
        height,
        channels,
        data,
        100);
}

