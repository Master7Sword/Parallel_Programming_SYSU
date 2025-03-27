# <center> 并行程序设计与算法实验2

## <center> 莫子昊 22331067

## 1.实验要求

> 使用MPI点对点通信方式实现并行通用矩阵乘法(MPI-v1)，并通过实验分析不同进程数量、矩阵规模时该实现的性能。

> 输入：m,n,k三个整数，每个整数的取值范围均为[128, 2048]
问题描述：随机生成m\*n的矩阵A及n\*k的矩阵B，并对这两个矩阵进行矩阵乘法运算，得到矩阵C.

> 输出：A,B,C三个矩阵，及矩阵计算所消耗的时间t。

> 要求：1. 使用MPI点对点通信实现并行矩阵乘法，调整并记录不同线程数量（1-16）及矩阵规模（128-2048）下的时间开销，填写下页表格，并分析其性能。

> 根据当前实现，在实验报告中讨论两个优化方向：a) 在内存有限的情况下，如何进行大规模矩阵乘法计算？b) 如何提高大规模稀疏矩阵乘法性能？

## 2.实验内容

首先，在Linux Ubuntu系统上配置MPI环境
```
apt-get install openmpi-bin libopenmpi-dev
```

MPI矩阵乘法代码如下，其中预留了命令行参数，方便后续使用脚本初始化矩阵尺寸m,n,k
```
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
```

然后使用脚本批量测试，并将运行时间记录在csv文件中：
```
OUTPUT_FILE="mpi_matrix_mul_results_$(date +%Y%m%d_%H%M%S).csv"

MATRIX_SIZES=(128 256 512 1024 2048)
PROCESSES=(1 2 4 8 16)

echo "Matrix Size,Processes,Time (s)" > "$OUTPUT_FILE"

for size in "${MATRIX_SIZES[@]}"; do
    for np in "${PROCESSES[@]}"; do
        echo "Testing size=$size with $np processes..."

        time_output=$(mpirun -np "$np" -host "localhost:$np" ./MPI "$size" "$size" "$size" 2>&1)
        
        exec_time=$(echo "$time_output" | grep "Matrix multiplication time" | awk '{print $4}')
        echo "$size,$np,$exec_time" >> "$OUTPUT_FILE"
        
        sleep 1
    done
done

echo "All tests completed! Results saved to $OUTPUT_FILE"
```

最后得到的运行时间如下

| 矩阵规模 \ 进程数 | 1           | 2           | 4           | 8           | 16          |
|------------------|-------------|-------------|-------------|-------------|-------------|
| 128              |0.002080     |0.001353     |0.001224     |0.001806     |0.149985     |
| 256              |0.024386     |0.014783     |0.008646     |0.007952     |0.292223     |
| 512              |0.244292     |0.101628     |0.115053     |0.070474     |0.260248     |
| 1024             |2.248341     |1.345771     |1.084222     |0.869982     |1.270133     |
| 2048             |65.620055    |36.690555    |25.239813    |19.854269    |21.713551    |

## 3.结果分析/讨论

进一步处理实验结果，得到相对1进程的加速比表格如下

| 矩阵规模 | 2进程加速比 | 4进程加速比 | 8进程加速比 | 16进程加速比 |
|----------|------------|------------|------------|-------------|
| 128      | 1.54x      | 1.70x      | 1.15x      | 0.01x       |
| 256      | 1.65x      | 2.82x      | 3.07x      | 0.08x       |
| 512      | 2.40x      | 2.12x      | 3.47x      | 0.94x       |
| 1024     | 1.67x      | 2.07x      | 2.58x      | 1.77x       |
| 2048     | 1.79x      | 2.60x      | 3.30x      | 3.02x       |

观察实验结果，可以发现：对于小矩阵(128,256)，使用16进程时出现严重性能下降（加速比<1），最佳进程数为4-8进程；
对于​中等矩阵(512,1024)，8进程时达到最佳加速，16进程时性能开始下降；
对于​大矩阵(2048)，随着进程数增加持续加速，16进程仍保持3x加速。

对于中小矩阵而言，16进程没能起到加速作用的原因是：通信启动时间远超过计算时间，且进程同步开销主导。
仅在2048矩阵上16进程仍保持加速，说明问题规模需足够大才能有效利用多进程。

### 讨论1：在内存有限的情况下，如何进行大规模矩阵乘法计算？

首先可以尝试分块计算：将矩阵划分为子块（block），每次只加载部分数据到内存计算。这样内存占用可控，适合GPU/CPU缓存优化。

其次可以采用​外存计算：使用磁盘存储矩阵数据，按需加载部分数据到内存（如HDF5格式），这样可以处理超大规模的矩阵运算。

最后也可以尝试​分布式计算：使用MPI将矩阵分块分布到多节点内存（如每个节点存储A的行块和B的列块）

但是以上的这些方法均会大幅增加通信的开销。

### 讨论2：如何提高大规模稀疏矩阵乘法性能？

可以采取稀疏存储，使用CSR/CSC/COO等格式存储非零元素（如SciPy的csr_matrix），减少内存占用。

或者利用GPU，使用CuSPARSE库的cusparseSpMM函数，相比CPU可以有将近百倍的加速。