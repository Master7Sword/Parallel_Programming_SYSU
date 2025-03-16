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
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    std::clock_t end = std::clock();

    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    printf("modified loop sequence: %.6lf seconds\n", elapsed_time);

    return 0;
}