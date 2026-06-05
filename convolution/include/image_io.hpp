#pragma once

unsigned char* loadImage(
    const char* filename,
    int& width,
    int& height,
    int& channels);

void saveImage(
    const char* filename,
    unsigned char* data,
    int width,
    int height,
    int channels);

