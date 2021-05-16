#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ 
void filter(int height, int width,  double *d_img, double *d_img_res)
{
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    int j = gidx / width / 3;
    int i = gidx / 3 - j * width;
    int ch = gidx - i * 3 - j * width * 3; 
    int size = height * width * 3;	
    double carr[121];

    if (gidx < size)
    {
        if (i < 4 || j < 4 || i > width - 5 || j > height - 5)
        {
            d_img_res[j*width*3 + i*3 + ch] =  d_img[j*width*3 + i*3 + ch];
        }
        else
        {
            int count = 0;
            for (int indi = -5; indi < 6; indi++)
            {
                for (int indj = -5; indj < 6; indj++)
                {
                    carr[count] = d_img[(j + indj)*width*3 + (i+indi)*3 + ch];
                    count++;
                }
            }
            double w = 0;
            for (int indi = 0; indi < 120; indi++)
            {
                for (int indj = indi + 1; indj < 121; indj++)
                {
                    if (carr[indi] < carr[indj])
                    {
                        w = carr[indi];
                        carr[indi] = carr[indj];
                        carr[indj] = w;
                    }
                }

            }
            d_img_res[j*width*3 + i*3 + ch] = carr[60];
        }
    }
}

int main(int argc, char **argv)
{
    int width, height, bpp, size;

    uint8_t* h_image = stbi_load("corgi.jpg", &width, &height, &bpp, 3);
    size = height * width * 3;
    double * h_buf = (double *) malloc(sizeof(double) * size);

    double *d_img, *d_img_res;
    cudaMalloc(&d_img, sizeof(double) * size);
    cudaMalloc(&d_img_res, sizeof(double) * size);

    for (int i = 0; i < size; i++)
    {
        h_buf[i] = (double) h_image[i];
    }

    cudaMemcpy(d_img, h_buf, sizeof(double) * size, cudaMemcpyHostToDevice);

    int block_size, grid_size;
    block_size = 1024;
    grid_size = size/block_size;
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    filter<<<dimGrid, dimBlock>>>(height, width, d_img, d_img_res);
    cudaDeviceSynchronize();

    double *h_buf_res = (double *)malloc(sizeof(double) * size);
    cudaMemcpy(h_buf_res, d_img_res, sizeof(double) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
        {
                h_image[i] = uint8_t (h_buf_res[i]);
        }
    stbi_write_png("res_corgi.jpg", width, height, 3, h_image, width * 3);
    return 0;
    }
