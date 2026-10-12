#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Функция умножения локальной части матрицы
void multiply(int *local_A, int *B, int *local_C, int rows, int M, int K) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < K; j++) {
            local_C[i * K + j] = 0;
            for (int k = 0; k < M; k++) {
                local_C[i * K + j] += local_A[i * M + k] * B[k * K + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 4, M = 4, K = 4; // Размеры матриц

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Число строк матрицы A должно делиться на количество процессов!\n");
        }
        MPI_Finalize();
        return 0;
    }

    int rows_per_proc = N / size; // Число строк на процесс

    // Выделяем память для локальных частей
    int *local_A = (int *)malloc(rows_per_proc * M * sizeof(int));
    int *B = (int *)malloc(M * K * sizeof(int));
    int *local_C = (int *)malloc(rows_per_proc * K * sizeof(int));

    // Инициализация матриц A и B в процессе 0
    int *A = NULL;
    int *C = NULL;
    if (rank == 0) {
        A = (int *)malloc(N * M * sizeof(int));
        C = (int *)malloc(N * K * sizeof(int));
        // Заполняем матрицы A и B
        for (int i = 0; i < N * M; i++) {
            A[i] = i + 1; // Пример заполнения матрицы A
        }
        for (int i = 0; i < M * K; i++) {
            B[i] = i + 1; // Пример заполнения матрицы B
        }
        printf("Матрица A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                printf("%d ", A[i * K + j]);
            }
            printf("\n");
        }
        printf("Матрица B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                printf("%d ", B[i * K + j]);
            }
            printf("\n");
        }
    }

    // Рассылаем строки матрицы A всем процессам
    MPI_Scatter(A, rows_per_proc * M, MPI_INT, local_A, rows_per_proc * M, MPI_INT, 0, MPI_COMM_WORLD);

    // Передача матрицы B каждому процессу вручную
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(B, M * K, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(B, M * K, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Умножение локальных частей
    multiply(local_A, B, local_C, rows_per_proc, M, K);

    // Сборка результата
    MPI_Gather(local_C, rows_per_proc * K, MPI_INT, C, rows_per_proc * K, MPI_INT, 0, MPI_COMM_WORLD);

    // Вывод результата в процессе 0
    if (rank == 0) {
        printf("Матрица C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                printf("%d ", C[i * K + j]);
            }
            printf("\n");
        }
        free(A);
        free(C);
    }

    // Освобождаем память
    free(local_A);
    free(B);
    free(local_C);

    MPI_Finalize();
    return 0;
}
