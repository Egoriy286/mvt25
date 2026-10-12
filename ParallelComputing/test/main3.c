#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define N 4

void multiply(int *local_A, int *B_T, int *local_C, int rows, int M, int K) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < K; j++) {
            local_C[i * K + j] = 0;
            for (int k = 0; k < M; k++) {
                local_C[i * K + j] += local_A[i * M + k] * B_T[j * M + k];
            }
        }
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int column_per_process = N / size;
    int extra_rows = N % size;

    int A[N][N];   // Матрица A
    int B[N][N];   // Матрица B
    int B_T[N][N]; // Транспонированная матрица B
    int C[N][N];   // Матрица результата C
    int result[N][N] = {{0}};

    int local_rows = (rank < extra_rows) ? rows_per_process + 1 : rows_per_process;
    int local_A[local_rows][N];  // Локальная часть матрицы A
    int local_B_T[column_per_process][N]; // **Локальная часть транспонированной матрицы B**
    int local_C[local_rows][column_per_process];  // **Локальная часть матрицы C**

    if(rank == 0 ) {
        A[0][0] = 1; A[0][1] = 2; A[1][0] = 3; A[1][1] = 4;
        B[0][0] = 1; B[0][1] = 2; B[1][0] = 3; B[1][1] = 4;
        // Транспонируем B
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B_T[j][i] = B[i][j];
            }
        }
    }

    // Раздаем строки матрицы A процессам
    MPI_Scatter(A, rows_per_process * N, MPI_INT, &local_A, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    // **Раздаем столбцы транспонированной матрицы B_T процессам**
    int cols_per_process = N / size;
    int count = cols_per_process * N;
    MPI_Scatter(B_T, count, MPI_INT, &local_B_T, count, MPI_INT, 0, MPI_COMM_WORLD);

    multiply(&local_A[0][0], &local_B_T[0][0], &local_C[0][0], local_rows, N, cols_per_process);

    // **Собрать результаты правильно**
    //MPI_Gather(&local_C, local_rows * cols_per_process, MPI_INT, &result, local_rows * cols_per_process, MPI_INT, 0, MPI_COMM_WORLD);
    int gather_cnts[size];
    int gather_displs[size];
    for (int i = 0; i < size; i++) {
        gather_cnts[i] = (i < extra_rows) ? (rows_per_process + 1) * cols_per_process : rows_per_process * cols_per_process;
        gather_displs[i] = (i == 0) ? 0 : (gather_displs[i - 1] + gather_cnts[i - 1]);
    }

MPI_Gatherv(&local_C[0][0], gather_cnts[rank], MPI_INT, &result, gather_cnts, gather_displs, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Результат умножения A*B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
        printf("Время выполнения: %f секунд \n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
