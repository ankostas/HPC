#include <stdio.h>

#include <math.h>

const int N = 128;

__global__ 
void laplacian(int n, double *da, double *dres)
{
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
        int x = tid / n;
        int y = tid - n * x;
    
        double x_l, x_r, y_l, y_r; 
        
        if (tid < n*n) 
        {
            y_l = y - 1 < 0 ? 0 : da[y - 1 + x*n];
            x_l = x - 1 < 0 ? 0 : da[y + (x - 1)*n];
            y_r = y + 1 > n - 1 ? 0 : da[y + 1 + x*n];
            x_r = x + 1 > n - 1 ? 0 : da[y + (x + 1)*n];

            dres[y + x*n] = (x_l + y_l + x_r + y_r) / 4.;

            if (x == 0 || y == n - 1 || x == n - 1) 
            {
                dres[y + x*n] = 0;
            }
        }

        __syncthreads();
        if (y == 0) dres[x*n] = 1;

        __syncthreads();
        da[y + x*n] = dres[y + x*n];      

}

void toFile(FILE *f_name, double *res_i, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(f_name, "%f\t", res_i[i*N + j]);
        }
    }
    fprintf(f_name, "\n");
}

int main()
{
    double *da;
    double *dres;
    cudaMalloc(&da, sizeof(double) * N * N);
    cudaMalloc(&dres, sizeof(double) * N * N);
    
    FILE *f_name;
    f_name = fopen("res.txt", "w");

    double *ha = (double *) calloc(sizeof(double), N * N);
    for (int i = 0; i < N; i++)
    {
        ha[i*N] = 1;
    }

    cudaMemcpy(da, ha, sizeof(double) * N * N, cudaMemcpyHostToDevice);

    dim3 dimBlock(1024);
    dim3 dimGrid(N * N / 1024);

    for (int k = 0; k < 200; k++)
    {
        laplacian<<<dimGrid, dimBlock>>>(N, da, dres);
        cudaMemcpy(ha, da, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
        toFile(f_name, ha, N);
    }
    return 0;
}
