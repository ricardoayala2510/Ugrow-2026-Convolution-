#pragma once

enum FilterType
{
    FILTER_BLUR = 0,
    FILTER_SHARPEN = 1,
    FILTER_EDGE_DETECTION = 2,
    FILTER_EMBOSS = 3
};

void launchConvolution(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    FilterType filter);
