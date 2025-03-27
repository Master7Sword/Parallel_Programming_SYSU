#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0

void initialize_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

void parallel_matrix_multiply(int rank, int size, double *A, double *B, double *C, int m, int n, int k) {
    int rows_per_process = m / size;
    int extra_rows = m % size;
    
    int local_rows = (rank < extra_rows) ? rows_per_process + 1 : rows_per_process;
    int local_m = local_rows;

    double *local_A = (double *)malloc(local_m * n * sizeof(double));
    double *local_C = (double *)malloc(local_m * k * sizeof(double));
    
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < extra_rows) ? (rows_per_process + 1) * n : rows_per_process * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                local_A, local_m * n, MPI_DOUBLE, 
                MASTER, MPI_COMM_WORLD);
    
    MPI_Bcast(B, n * k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < k; j++) {
            local_C[i * k + j] = 0.0;
            for (int l = 0; l < n; l++) {
                local_C[i * k + j] += local_A[i * n + l] * B[l * k + j];
            }
        }
    }
    
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *recvdispls = (int *)malloc(size * sizeof(int));
    
    offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_rows) ? (rows_per_process + 1) * k : rows_per_process * k;
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }
    
    MPI_Gatherv(local_C, local_m * k, MPI_DOUBLE, 
               C, recvcounts, recvdispls, MPI_DOUBLE, 
               MASTER, MPI_COMM_WORLD);
    
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int m, n, k;
    double *A = NULL, *B = NULL, *C = NULL;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == MASTER) {
            printf("Usage: mpirun -np <num_processes> %s <m> <n> <k>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    
    if (rank == MASTER) {
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));
        
        srand(time(NULL));
        initialize_matrix(A, m, n);
        initialize_matrix(B, n, k);
        
    } else {
        B = (double *)malloc(n * k * sizeof(double));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    parallel_matrix_multiply(rank, size, A, B, C, m, n, k);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == MASTER) {
        
        printf("\nMatrix multiplication time: %.6f seconds\n", end_time - start_time);

        free(A);
        free(B);
        free(C);
    } else {
        free(B);
    }
    
    MPI_Finalize();
    return 0;
}