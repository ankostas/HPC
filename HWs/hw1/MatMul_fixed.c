#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}

int main()
{
    const size_t N = 1000; // size of an array

    clock_t start, end;   
 
    double ** A, ** B, ** C; // matrices

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);
    
    int i, j, n;
    
    start = clock();
    #pragma omp parallel 
    {
    #pragma omp for schedule(static) collapse(2) private(n)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (n = 0; n < N; n++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (ijn) parallelization: %.1f seconds.\n", (double)(end - start));
    zero_init_matrix(C, N);
    
    start = clock();
    {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int n = 0; n < N; n++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (ijn): %.1f seconds.\n", (double)(end - start));
    zero_init_matrix(C, N);
    
    start = clock();
    #pragma omp parallel
    { 
    #pragma omp for collapse(2) schedule(static) private(n)
    for (j = 0; j < N; j++)
    {
        for (i = 0; i < N; i++)
        {
            for (n = 0; n < N; n++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (jin) parallelication: %.1f seconds.\n", (double)(end - start));
    zero_init_matrix(C, N);
    
    start = clock();
    {
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int n = 0; n < N; n++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (jin): %.1f seconds.\n", (double)(end - start) );   
    
    start = clock();   
    #pragma omp parallel
    { 
    #pragma omp for private(n, i, j) schedule(static)
    for (n = 0; n < N; n++)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (nij) parallelization: %.1f seconds.\n", (double)(end - start)); 
    zero_init_matrix(C, N);
    
    start = clock();
    {
    for (int n = 0; n < N; n++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }
    }
    end = clock();
    printf("Time elapsed (nij): %.1f seconds.\n", (double)(end - start) );
    zero_init_matrix(C, N);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}