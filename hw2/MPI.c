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
    
    // 分发A矩阵
    if (rank == MASTER) {
        int offset = 0;
        for (int dest = 0; dest < size; dest++) {
            int dest_rows = (dest < extra_rows) ? rows_per_process + 1 : rows_per_process;
            if (dest == MASTER) {
                for (int i = 0; i < dest_rows * n; i++) {
                    local_A[i] = A[offset + i];
                }
            } else {
                MPI_Send(A + offset, dest_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
            offset += dest_rows * n;
        }
    } else {
        MPI_Recv(local_A, local_m * n, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // 广播B矩阵
    if (rank == MASTER) {
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(B, n * k, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(B, n * k, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // 本地计算
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < k; j++) {
            local_C[i * k + j] = 0.0;
            for (int l = 0; l < n; l++) {
                local_C[i * k + j] += local_A[i * n + l] * B[l * k + j];
            }
        }
    }
    
    // 收集结果
    if (rank == MASTER) {
        int offset = 0;
        for (int src = 0; src < size; src++) {
            int src_rows = (src < extra_rows) ? rows_per_process + 1 : rows_per_process;
            if (src == MASTER) {
                for (int i = 0; i < src_rows * k; i++) {
                    C[offset + i] = local_C[i];
                }
            } else {
                MPI_Recv(C + offset, src_rows * k, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            offset += src_rows * k;
        }
    } else {
        MPI_Send(local_C, local_m * k, MPI_DOUBLE, MASTER, 2, MPI_COMM_WORLD);
    }
    
    free(local_A);
    free(local_C);
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