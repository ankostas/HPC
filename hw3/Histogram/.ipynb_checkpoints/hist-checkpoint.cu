#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__global__ 
void Blur(int height, int width,  uint8_t *d_img, unsigned int *d_res)
{
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;
    int size = height * width;
    if (gidx < size)
    {
        unsigned char value = d_img[gidx];
        int bin = value % 256;
        atomicAdd(&d_res[bin], 1);
    }
}

int main(int argc, char **argv)
{
    int width, height, bpp, size;
    FILE *fp;
    fp = fopen("res.txt", "w");

    uint8_t* h_img_0 = stbi_load("corgi.jpg", &width, &height, &bpp, 3);
    size = height * width;

    uint8_t* h_img = (uint8_t *) malloc(sizeof(uint8_t) * size);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            h_img[j*width + i] = (h_img_0[j*width*3 + i*3] + \
                        h_img_0[j*width*3 + i*3 + 1] + \
                        h_img_0[j*width*3 + i*3 + 2]) / 3.;
        }
    }

    uint8_t *d_img;
    unsigned int *d_res;
    unsigned int *h_res = (unsigned int *) malloc(sizeof(unsigned int) * 256);
    cudaMalloc(&d_img, sizeof(uint8_t) * size);
    cudaMalloc(&d_res, sizeof(unsigned int) * 256);
    cudaMemset(d_res, 0, sizeof(unsigned int) * 256);
    cudaMemcpy(d_img, h_img, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);

    int block_size, grid_size;
    block_size = 256;
    grid_size = size / block_size;
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    Blur<<<dimGrid, dimBlock>>>(height, width, d_img, d_res);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; i++)
    {
        fprintf(fp, "%d\t", h_res[i]);
    }
    return 0;
}
