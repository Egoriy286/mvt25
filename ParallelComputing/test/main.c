#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = argc > 1 ? atoi(argv[1]) : 1000;

    if (N <= 0 || N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "N должно быть положительным и кратным size\n");
        MPI_Finalize();
        return 1;
    }

    int rows = N / size;

    double *A = NULL;
    double *x = malloc(N * sizeof(double));
    double *local_A = malloc(rows * N * sizeof(double));
    double *local_y = calloc(rows, sizeof(double));
    double *y = NULL;

    if (!x || !local_A || !local_y) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = malloc((size_t)N * N * sizeof(double));
        y = malloc(N * sizeof(double));

        if (!A || !y)
            MPI_Abort(MPI_COMM_WORLD, 1);

        for (int i = 0; i < N; i++) {
            x[i] = i + 1.0;

            for (int j = 0; j < N; j++)
                A[i * N + j] = 1.0 / (i + j + 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Передаём вектор всем процессам
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределяем строки матрицы
    MPI_Scatter(
        A, rows * N, MPI_DOUBLE,
        local_A, rows * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Локальное умножение
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++)
            local_y[i] += local_A[i * N + j] * x[j];

    // Собираем результат
    MPI_Gather(
        local_y, rows, MPI_DOUBLE,
        y, rows, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    double elapsed = MPI_Wtime() - t0;
    double max_time;

    MPI_Reduce(
        &elapsed, &max_time, 1,
        MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("N = %d, Processes = %d, Time = %.6f s\n",
               N, size, max_time);

        FILE *file = fopen("result.txt", "w");
        if (file) {
            for (int i = 0; i < N; i++)
                fprintf(file, "%d %.12f\n", i, y[i]);
            fclose(file);
        }
    }

    free(A);
    free(x);
    free(y);
    free(local_A);
    free(local_y);

    MPI_Finalize();
    return 0;
}