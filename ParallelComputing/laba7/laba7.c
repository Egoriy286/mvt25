#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define K 0.1
#define TOL 1e-10
#define MAX_ITER 2000000
#define CHECK_EVERY 50

typedef struct {
    double left, diagonal, right, b;
} Row;

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    if (rank == 0 && (argc == 2 || argc == 3)) {
        char *end;
        errno = 0;
        long cells = strtol(argv[1], &end, 10);
        if (errno == 0 && *end == '\0' && cells >= 2 && cells <= INT_MAX) {
            N = (int)cells;
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N < 2 || size >= N) {
        if (rank == 0) {
            fprintf(stderr, "Запуск: %s N [решение.csv], где N больше числа процессов.\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int inner = N - 1;
    int rows = inner / size + (rank < inner % size);
    int first = 1 + rank * (inner / size) + (rank < inner % size ? rank : inner % size);
    int left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int right = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    Row *A = malloc((size_t)rows * sizeof(*A));
    double *u = calloc((size_t)rows + 2, sizeof(*u));
    double *u_new = calloc((size_t)rows + 2, sizeof(*u_new));
    if (!A || !u || !u_new) {
        fprintf(stderr, "Процесс %d: не удалось выделить память.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    // Каждый процесс заполняет три диагонали только своих строк.
    double h = 1.0 / N;
    double coefficient = K / (h * h);
    for (int i = 0; i < rows; i++) {
        int node = first + i;
        A[i] = (Row){node > 1 ? -coefficient : 0.0, 2.0 * coefficient,
                     node < inner ? -coefficient : 0.0, 1.0};
    }

    double residual = 1.0;
    int iter = 0;
    for (; iter <= MAX_ITER; iter++) {
        // В двух добавочных ячейках храним значения с соседних процессов.
        if (size > 1) {
            MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0, &u[rows + 1], 1,
                         MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[rows], 1, MPI_DOUBLE, right, 1, &u[0], 1,
                         MPI_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (iter % CHECK_EVERY == 0 || iter == MAX_ITER) {
            double local_sum = 0.0, global_sum;
            for (int i = 1; i <= rows; i++) {
                Row row = A[i - 1];
                double r = row.b - (row.left * u[i - 1] + row.diagonal * u[i]
                                    + row.right * u[i + 1]);
                local_sum += r * r;
            }
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            residual = sqrt(global_sum / inner); // ||Au-b||_2 / ||b||_2, где b_i = 1
            if (residual <= TOL || iter == MAX_ITER) {
                break;
            }
        }

        for (int i = 1; i <= rows; i++) {
            Row row = A[i - 1];
            u_new[i] = (row.b - row.left * u[i - 1] - row.right * u[i + 1]) / row.diagonal;
        }
        double *old = u;
        u = u_new;
        u_new = old;
    }

    double elapsed = MPI_Wtime() - start;
    double time;
    MPI_Reduce(&elapsed, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_error = 0.0, max_error;
    for (int i = 1; i <= rows; i++) {
        double x = (first + i - 1) * h;
        double exact = x * (1.0 - x) / (2.0 * K);
        local_error = fmax(local_error, fabs(u[i] - exact));
    }
    MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = residual <= TOL ? EXIT_SUCCESS : EXIT_FAILURE;
    if (rank == 0 && status != EXIT_SUCCESS) {
        fprintf(stderr, "Метод Якоби достиг предела %d итераций; невязка %.3e.\n", iter, residual);
    }
    if (status == EXIT_SUCCESS && argc == 3) {
        double *solution = NULL;
        int *counts = NULL, *offsets = NULL;
        if (rank == 0) {
            solution = malloc((size_t)inner * sizeof(*solution));
            counts = malloc((size_t)size * sizeof(*counts));
            offsets = malloc((size_t)size * sizeof(*offsets));
            if (!solution || !counts || !offsets) {
                fprintf(stderr, "Не удалось выделить память для сборки решения.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return EXIT_FAILURE;
            }
            int offset = 0;
            for (int p = 0; p < size; p++) {
                counts[p] = inner / size + (p < inner % size);
                offsets[p] = offset;
                offset += counts[p];
            }
        }
        MPI_Gatherv(&u[1], rows, MPI_DOUBLE, solution, counts, offsets,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            FILE *file = fopen(argv[2], "w");
            if (file) {
                fputs("x,u\n0,0\n", file);
                for (int i = 1; i < N; i++) {
                    fprintf(file, "%.17g,%.17g\n", i * h, solution[i - 1]);
                }
                fputs("1,0\n", file);
                int write_error = ferror(file);
                if (fclose(file) != 0 || write_error) {
                    fprintf(stderr, "Не удалось записать %s.\n", argv[2]);
                    status = EXIT_FAILURE;
                }
            } else {
                perror(argv[2]);
                status = EXIT_FAILURE;
            }
        }
        free(offsets);
        free(counts);
        free(solution);
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0 && status == EXIT_SUCCESS) {
        printf("N=%d processes=%d time_s=%.9f iterations=%d residual=%.6e max_error=%.6e\n",
               N, size, time, iter, residual, max_error);
    }
    free(u_new);
    free(u);
    free(A);
    MPI_Finalize();
    return status;
}
