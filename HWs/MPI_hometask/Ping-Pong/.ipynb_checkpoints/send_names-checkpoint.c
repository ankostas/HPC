#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <string.h>

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
    
    int N = 6;
        
    int number, f=0, count;
    int len_init, len_cur;
    int sender = 0, reciever;

    char rank_buff[5];
    char *name = (char *)malloc(sizeof(char));
    char *cur_name = (char *)malloc(sizeof(char));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    sprintf(rank_buff, "%d", prank);
    cur_name = concat((char *)argv[1], rank_buff);
    len_init=strlen(cur_name);

    while(1)
    {
        if (prank == sender)
        {
            do
            {
                reciever = rand() % psize;
            } while(sender == reciever);
            printf("Process %d, send data to %d\n", sender, reciever);
            MPI_Ssend(cur_name, strlen(cur_name) + 1, MPI_CHAR, reciever, 0, MPI_COMM_WORLD);
                sender = reciever;
        }
        else
        {
            while(!f)
            {
                MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &f, &status);
                sleep(1);
            }
            MPI_Get_count(&status, MPI_CHAR, &count);
            char *name = (char *)malloc(sizeof(char)*(count + 1));
            MPI_Recv(name, count + 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            cur_name = concat(name, cur_name);
            printf("Process %d received message %s\n", prank, name);
            f = 0;
            sender = prank;
            len_cur=strlen(name);
            }

            if (len_cur / len_init  == N)
            {
                        MPI_Abort(MPI_COMM_WORLD, 911);
                        break;
            }

    }
    return 0;
    }
