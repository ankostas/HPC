#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> 

char* concat(char *s1, char *s2)
{
    char *res =(char *) malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(res, s1);
    strcat(res, s2);
    return res;
}

int main(int argc, char ** argv)
{
    int psize, prank;
    MPI_Status status;
    MPI_Request request;

    int f=0, cnt=0;
    int N = 8;
    int sender = 0;
    int reciever;
    
    char rank_buff[5];
       
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    
    char *name;
    name = concat(argv[1],argv[1]);

    double start, end;
    start = MPI_Wtime();
    while (1)
    {
        if (prank == sender)
        {
            do
            {
                reciever = rand() % psize;
            } while(sender == reciever);
            
            MPI_Ssend(name, strlen(name) + 1, MPI_CHAR, reciever, 0, MPI_COMM_WORLD);
            MPI_Send(&cnt, 1, MPI_INT, reciever, 1, MPI_COMM_WORLD);
            sender = reciever;
        }
        else
        {
            while(!f)
            {
                MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &f, &status);
            }
            MPI_Recv(name, strlen(name) + 1, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&cnt, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt++;
            f = 0;
            sender = prank;
            end = MPI_Wtime();
            if ((end - start) > 10)
            {
                printf("%lu\t%d\t%f\t%f\t%f\n", strlen(name), cnt, end - start, (double) (end - start) / (double) cnt, (double) cnt * strlen(name) / (end - start) / 1024. / 1024.) ;
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
        }
    }
    return 0;
    }
