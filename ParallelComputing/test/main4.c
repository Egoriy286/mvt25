#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4;  // Количество строк первой матрицы A и результата C
    const int M = 4;  // Количество столбцов первой матрицы A и строк второй матрицы B
    const int K = 4;  // Количество столбцов второй матрицы B и результата C

    if (N % size != 0) {
        if (rank == 0) {
            printf("Ошибка: количество строк N должно делиться нацело на количество процессов.\n");
        }
        MPI_Finalize();
        return -1;
    }

    int rows_per_process = N / size;  // Число строк на процесс

    // Матрицы
    int A[N][M];   // Матрица A
    int B[M][K];   // Матрица B
    int C[N][K];   // Матрица результата C

    // Локальные данные для каждого процесса
    int local_A[rows_per_process][M];  // Локальная часть матрицы A
    int local_C[rows_per_process][K];  // Локальная часть матрицы C

    // Инициализация матриц A и B на процессе 0
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                A[i][j] = i * M + j + 1;  // Заполнение матрицы A
            }
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                B[i][j] = i * K + j + 1;  // Заполнение матрицы B
            }
        }
    }

    // Передаем матрицу B на все процессы
    MPI_Bcast(&B, M * K, MPI_INT, 0, MPI_COMM_WORLD);

    // Раздаем строки матрицы A процессам
    MPI_Scatter(A, rows_per_process * M, MPI_INT, local_A, rows_per_process * M, MPI_INT, 0, MPI_COMM_WORLD);

    // Умножение локальных строк матрицы A на всю матрицу B
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < K; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < M; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Сбор результата в матрицу C
    MPI_Gather(local_C, rows_per_process * K, MPI_INT, C, rows_per_process * K, MPI_INT, 0, MPI_COMM_WORLD);

    // Печать результата на процессе 0
    if (rank == 0) {
        printf("Результат умножения матрицы A на матрицу B (матрица C):\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
