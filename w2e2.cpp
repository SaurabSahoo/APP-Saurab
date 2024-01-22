#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
        int my_rank;
        int p;
        MPI_Status status;

        int broadcast_integer = -1;
        int total_sources;
        int stage;
        int source;
        int dest;
        int tag = 0;
        // variables for algorithm
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        // broadcast algorithm starts
        if (my_rank == 0) broadcast_integer = 100;
        total_sources = 1;
        stage = 0;
        while(total_sources < p)
        {
                if(my_rank < total_sources && my_rank + total_sources < p)
                {
                        dest = my_rank + total_sources;
                        printf("%d sends to %d, stage %d \n", my_rank, dest, stage);
                        MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                }
                else if(my_rank < p && my_rank < 2*total_sources && my_rank > total_sources)
                {
                        source = my_rank - total_sources;
                        printf("%d receives from %d, stage %d \n", my_rank, source, stage);
                        MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
                }
                total_sources = 2*total_sources;
                stage++;
                // printf("hi from %d \n", my_rank);
        }
        // printf("exiting from %d ---------------------- \n", my_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank == 0)
                printf("Total steps: %d \n", stage);
        MPI_Finalize();
        return 0;
}