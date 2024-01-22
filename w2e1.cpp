#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int my_rank;            /* Process rank */
    int p;                  /* Total number of processes */
    int source;             /* Sender's rank */
    int dest;               /* Receiver's rank */
    int tag = 0;            /* Tag for messages */
    int broadcast_integer = -1; /* Integer for broadcast */
    int spacing, stage = 0; /* Broadcast stage */
    MPI_Status status;      /* Status for receive operation */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    broadcast_integer = -1;
    if (my_rank == 0) 
        broadcast_integer = 100;

    spacing = p;
    stage = 0;

    while (spacing > 1) {
        if (my_rank % spacing == 0) {
            dest = my_rank + spacing / 2;
            printf("%d sends to %d, Stage %d\n", my_rank, dest, stage);
            MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        } else if (my_rank % (spacing / 2) == 0) {
            source = my_rank - spacing / 2;
            printf("%d receives from %d, Stage %d\n", my_rank, source, stage);
            MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        }

        spacing = spacing / 2;
        stage = stage + 1;
    }

    printf("Process %d has the final broadcasted value: %d\n", my_rank, broadcast_integer);

    MPI_Finalize();
    
    return 0;
}
