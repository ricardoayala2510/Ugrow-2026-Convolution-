#pragma once

void launchBlur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY
);

void launchSharpen(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int blockX,
    int blockY
);
