#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    const int n = 1024; 
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(n));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

    std::srand(std::time(0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
            B[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    std::clock_t start = std::clock();

    const int unroll_factor = 4; // 展开因子
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double A_ik = A[i][k]; // 将 A[i][k] 提取出来，减少内存访问次数
            int j = 0;

            // 主循环展开
            for (; j <= n - unroll_factor; j += unroll_factor) {
                C[i][j] += A_ik * B[k][j];
                C[i][j + 1] += A_ik * B[k][j + 1];
                C[i][j + 2] += A_ik * B[k][j + 2];
                C[i][j + 3] += A_ik * B[k][j + 3];
            }

            // 处理剩余的部分
            for (; j < n; ++j) {
                C[i][j] += A_ik * B[k][j];
            }
        }
    }

    std::clock_t end = std::clock();

    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    printf("modified loop sequence and unrolled: %.6lf seconds\n", elapsed_time);

    return 0;
}