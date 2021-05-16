#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ 
void filter(int height, int width, double *ker, double *d_img, double *d_img_res)
{
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    int j = gidx / width / 3;
    int i = gidx / 3 - j * width;
    int ch = gidx - i * 3 - j * width * 3; 
    int size = height * width * 3;
    if (gidx < size)
    {
        if (i == 0 || j == 0 || i == width - 1 || j == height - 1)
        {
            d_img_res[j*width*3 + i*3 + ch] =  d_img[j*width*3 + i*3 + ch];
        }
        else
        {
            d_img_res[j*width*3 + i*3 + ch] =  (d_img[j*width*3 + i*3 + ch]*ker[4] + \
                                                    d_img[(j + 1) *width * 3 + (i - 1) * 3 + ch]*ker[0] + \
                                                    d_img[(j + 1) *width * 3 + (i + 1) * 3 + ch]*ker[8] + \
                                                    d_img[(j - 1) *width * 3 + (i - 1) * 3 + ch]*ker[6] + \
                                                    d_img[(j - 1) *width * 3 + (i + 1) * 3 + ch]*ker[2] + \
                                                    d_img[(j + 1) *width * 3 + i * 3 + ch]*ker[3] + \
                                                    d_img[j *width * 3 + (i - 1) * 3 + ch]*ker[1] + \
                                                    d_img[(j - 1) *width * 3 + i * 3 + ch]*ker[5] + \
                                                    d_img[j * width * 3 + (i + 1)*3 + ch]*ker[7]); 
        }
        if (d_img_res[j*width*3 +i*3 + ch] < 0)
        {
            d_img_res[j*width*3 + i*3 + ch] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    int width, height, bpp, size;
    double *ker = (double *) calloc(sizeof(double), 9);
    double *d_ker;

    char *ker_name;
    ker_name = argv[1];

    if (strcmp(ker_name, "edge") == 0)
    {
        ker[0] = ker[6] = ker[2] = ker[8] = -1;
        ker[1] = ker[3] = ker[7] = ker[5] = -1;
        ker[4] = 8;
    }
    else if (strcmp(ker_name, "sharpen") == 0)
    {
        ker[0] = ker[6] = ker[2] = ker[8] = 0;
        ker[1] = ker[3] = ker[7] = ker[5] = -1;
        ker[4] = 5;
    }
    else if (strcmp(ker_name, "gaussian") == 0)
    {
        ker[0] = ker[6] = ker[2] = ker[8] = 1 / 16.;
        ker[1] = ker[3] = ker[7] = ker[5] = 2 / 16.;
        ker[4] = 4 / 16.;
    }

    cudaMalloc(&d_ker, sizeof(double)*9);
    cudaMemcpy(d_ker, ker, sizeof(double) * 9, cudaMemcpyHostToDevice);

    uint8_t* h_img = stbi_load("corgi.jpg", &width, &height, &bpp, 3);
    size = height * width * 3;
    double * h_buf = (double *) malloc(sizeof(double) * size);

    double *d_img;
    double *d_img_res;
    cudaMalloc(&d_img, sizeof(double) * size);
    cudaMalloc(&d_img_res, sizeof(double) * size);

    for (int i = 0; i < size; i++)
    {
        h_buf[i] = (double) h_img[i];
    }
    cudaMemcpy(d_img, h_buf, sizeof(double) * size, cudaMemcpyHostToDevice);
    int block_size, grid_size;
    block_size = 1024;
    grid_size = size/block_size;
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    filter<<<dimGrid, dimBlock>>>(height, width, d_ker, d_img, d_img_res);
    cudaDeviceSynchronize();

    double *h_buf_res = (double *)malloc(sizeof(double) * size);
    cudaMemcpy(h_buf_res, d_img_res, sizeof(double) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
        {
                h_img[i] = uint8_t (h_buf_res[i]);
        }
    stbi_write_png("res_corgi.jpg", width, height, 3, h_img, width * 3);
    return 0;
}
