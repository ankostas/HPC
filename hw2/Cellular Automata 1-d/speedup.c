#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

void create_mask(int *mask, int dec)
{
    int i = 0;
    while(dec != 0)
    {
        if (dec % 2 == 1)
        {
            mask[7 - i] = 1;
        }
        dec = dec / 2;
        i++;
    }
}

void step_const(int *a, int N, int* mask, int* next)
{
    int ind;
    next[0] = a[0];
    next[N - 1] = a[N - 1];
    for (int i = 1; i < (N - 1); i++)
    {
        ind = 7 - a[i - 1]*4 - a[i]*2 - a[i + 1];
        next[i] = mask[ind];
    }
    for (int i = 0; i < N; i++)
    {
        a[i] = next[i];
    }
}

int main(int argc, char** argv)
{
    int N, dec, n_steps = 1000;    
    int psize, prank;
    MPI_Status status;
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    
    dec = atoi(argv[1]);
    N = atoi(argv[2]);  
    
    int *a = (int *)calloc(sizeof(int), N);
    int *next = (int *)malloc(sizeof(int)*N);
    int *mask = (int *)calloc(sizeof(int), 8);
    create_mask(mask, dec);
    int cur_left = N / psize * prank;
    int cur_right =  N / psize * (prank + 1);
    int *size = (int *)calloc(sizeof(int), psize);
    int *displs = (int *)calloc(sizeof(int), psize);

    double start;
    for (int k = 0; k < psize; k++)
    {
        size[k] = (int) N / (double) psize;
        displs[k] = k * size[k];
    }
    size[psize - 1] += N % psize;
    if (prank == (psize - 1))
    {
        cur_right = N; 
    }
    int right = (prank + 1) % psize;
    int left = (prank - 1) < 0 ? psize - 1 : prank - 1;
    int cur_size = cur_right - cur_left + 2;
    int* cur_a = (int *)calloc(sizeof(int), cur_size);
    int* cur_next = (int *)malloc(sizeof(int) * cur_size);
    int* cur_a_to_send = (int *)malloc(sizeof(int) * (cur_size - 2));

    unsigned int seed = (prank + 1)*42434;
    for (int i = 0; i < cur_size; i++)
    {
        cur_a[i] = rand_r(&seed) % 2;
    }
    if (prank == 0)
    {
        start = MPI_Wtime();
        int *a = (int *)malloc(sizeof(int) * N);
    }
    for (int i = 0; i < n_steps; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(&cur_a[1], 1, MPI_INT, left, 0, MPI_COMM_WORLD);
        MPI_Send(&cur_a[cur_size - 2], 1, MPI_INT, right, 1, MPI_COMM_WORLD);
        MPI_Recv(&cur_a[cur_size - 1], 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cur_a[0], 1, MPI_INT, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        step_const(cur_a, cur_size, mask, cur_next);
    }
    if (prank == 0)
    {
        printf("%d\t%f\n", psize, (double) (MPI_Wtime() - start));
    }
    return 0;
}
