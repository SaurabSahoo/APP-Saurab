#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void Parallel_matrix_vector_prod(float* local_A, int m, int n, float* local_x, float* global_x, float* local_y, int local_m, int local_n) {
    int i, j;

    MPI_Allgather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, MPI_COMM_WORLD);

    for (i = 0; i < local_m; i++) {
        local_y[i] = 0.0;
        for (j = 0; j < n; j++) {
            local_y[i] += local_A[i * n + j] * global_x[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int my_rank, p;
    float* local_A;
    float* global_x;
    float* local_x;
    float* local_y;
    float* temp;
    int i, j;
    int m, n;
    int local_m, local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        printf("Enter dimensions of the array mXn\n");
        scanf("%d %d", &m, &n);
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_m = m / p;
    local_n = n / p;

    local_A = (float*)malloc((local_m * n) * sizeof(float));
    local_x = (float*)malloc(local_n * sizeof(float));
    local_y = (float*)malloc(local_m * sizeof(float));
    global_x = (float*)malloc(n * sizeof(float));

    temp = (float*)malloc((p * local_m * n) * sizeof(float));

    if (my_rank == 0) {
        printf("\nEnter elements of the array row-wise\n");
        for (i = 0; i < p * local_m * n; i++) {
            scanf("%f", &temp[i]);
        }
    }

    MPI_Scatter(temp, local_m * n, MPI_FLOAT, local_A, local_m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Gather(local_A, local_m * n, MPI_FLOAT, temp, local_m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("MATRIX:\n");
        for (i = 0; i < p * local_m; i++) {
            for (j = 0; j < n; j++) {
                printf("%4.1f ", temp[i * p * local_m + j]);
            }
            printf("\n");
        }
    }

    free(temp);
    temp = (float*)malloc((p * local_n) * sizeof(float));

    if (my_rank == 0) {
        printf("\nEnter elements of the vector\n");
        for (i = 0; i < p * local_n; i++) {
            scanf("%f", &temp[i]);
        }
    }

    MPI_Scatter(temp, local_n, MPI_FLOAT, local_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(temp);
    temp = (float*)malloc((p * local_n) * sizeof(float));
    MPI_Gather(local_x, local_n, MPI_FLOAT, temp, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Vector:\n");
        for (i = 0; i < p * local_n; i++) {
            printf("%4.1f ", temp[i]);
        }
        printf("\n");
    }

    Parallel_matrix_vector_prod(local_A, m, n, local_x, global_x, local_y, local_m, local_n);

    free(temp);
    temp = (float*)malloc((p * local_m) * sizeof(float));
    MPI_Gather(local_y, local_m, MPI_FLOAT, temp, local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Resulting Vector:\n");
        for (i = 0; i < p * local_m; i++) {
            printf("%4.1f ", temp[i]);
        }
        printf("\n");
    }

    free(temp);
    free(local_A);
    free(local_x);
    free(local_y);
    free(global_x);

    MPI_Finalize();
    return 0;
}
