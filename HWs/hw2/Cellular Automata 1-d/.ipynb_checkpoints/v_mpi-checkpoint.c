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

void matrix_print(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d\t", a[i]);
        if ((i + 1) % 10 == 0 || i == (N - 1))
        {
            printf("\n");
        }
    }
}

void matrix_print_file(int *matrix, int N, FILE *f)
{
    for (int i = 0; i < N; i++)
    {
        fprintf(f, "%d\t", matrix[i]);
    }
    fprintf(f, "\n");
}

int main(int argc, char** argv)
{
    int N = 100, n_steps = 100;
    
    int *a = (int *)calloc(sizeof(int), N);
    int *next = (int *)malloc(sizeof(int)*N);
    int *mask = (int *)calloc(sizeof(int), 10);

    int psize, prank;
    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    int dec;
    dec = atoi(argv[1]);

    char *condition;
    condition = argv[2];
    
    FILE *f;
    f = fopen("res.txt", "a+");

    if (prank == 0)
    {
        int *a = (int *)malloc(sizeof(int) * N);
    }
    create_mask(mask, dec);
    int cur_left = N / psize * prank;
    int cur_right =  N / psize * (prank + 1);
    int *size = (int *)calloc(sizeof(int), psize);
    int *displs = (int *)calloc(sizeof(int), psize);
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
    printf("%d\t", cur_size - 2);
    int* cur_a = (int *)calloc(sizeof(int), cur_size);
    int* cur_next = (int *)malloc(sizeof(int) * cur_size);
    int* cur_a_to_send = (int *)malloc(sizeof(int) * (cur_size - 2));

    unsigned int seed = (prank + 1)*42434;
    for (int i = 0; i < cur_size; i++)
    {
        cur_a[i] = rand_r(&seed) % 2;
    }

    for (int i = 0; i < n_steps; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(&cur_a[1], 1, MPI_INT, left, 0, MPI_COMM_WORLD);
        MPI_Send(&cur_a[cur_size - 2], 1, MPI_INT, right, 1, MPI_COMM_WORLD);
        MPI_Recv(&cur_a[cur_size - 1], 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cur_a[0], 1, MPI_INT, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        step_const(cur_a, cur_size, mask, cur_next);
        if (i % 1 == 0)
        {
            for (int j = 1; j < (cur_size - 1); j++)
            {
                cur_a_to_send[j - 1] = cur_a[j];
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Gatherv(cur_a_to_send, (cur_size - 2), MPI_INT, a, size, displs, MPI_INT, 0, MPI_COMM_WORLD);
            if (prank == 0)
            {
                matrix_print_file(a, N, f);
            }
        }
    }
    return 0;
}
