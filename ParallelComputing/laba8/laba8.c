#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция умножения локальной части матрицы
static void multiply(const int *local_A, const int *B, int *local_C, int rows, int N) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }
}

static int save_matrix(const char *name, const int *matrix, int N) {
    FILE *file = fopen(name, "w");
    if (!file) {
        perror(name);
        return 0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%d%c", matrix[i * N + j], j + 1 == N ? '\n' : ' ');
        }
    }
    int write_error = ferror(file);
    if (fclose(file) != 0 || write_error) {
        fprintf(stderr, "Не удалось записать %s.\n", name);
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    if (rank == 0 && argc == 2) {
        char *end;
        errno = 0;
        long length = strtol(argv[1], &end, 10);
        if (errno == 0 && *end == '\0' && length > 0 && length <= INT_MAX / length) {
            N = (int)length;
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N == 0) {
        if (rank == 0) {
            fprintf(stderr, "Запуск: %s N, где N > 0 и N * N <= %d.\n", argv[0], INT_MAX);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = N / size;
    int extra_rows = N % size;
    int rows = rows_per_proc + (rank < extra_rows);
    int local_count = rows * N;
    int count = N * N;

    // Выделяем хотя бы один элемент и на процессах без строк.
    int *local_A = calloc(local_count ? local_count : 1, sizeof(*local_A));
    int *local_C = calloc(local_count ? local_count : 1, sizeof(*local_C));
    int *B = calloc(count, sizeof(*B));
    int *A = NULL, *C = NULL, *counts = NULL, *offsets = NULL;
    if (rank == 0) {
        A = calloc(count, sizeof(*A));
        C = calloc(count, sizeof(*C));
        counts = calloc(size, sizeof(*counts));
        offsets = calloc(size, sizeof(*offsets));
    }
    if (!local_A || !local_C || !B || (rank == 0 && (!A || !C || !counts || !offsets))) {
        fprintf(stderr, "Процесс %d не смог выделить память.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        srand((unsigned int)time(NULL));
        for (int i = 0; i < count; i++) {
            A[i] = rand() % 21;
            B[i] = rand() % 21;
        }
        int offset = 0;
        for (int p = 0; p < size; p++) {
            counts[p] = (rows_per_proc + (p < extra_rows)) * N;
            offsets[p] = offset;
            offset += counts[p];
        }
    }

    MPI_Scatterv(A, counts, offsets, MPI_INT, local_A, local_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, count, MPI_INT, 0, MPI_COMM_WORLD);
    multiply(local_A, B, local_C, rows, N);
    MPI_Gatherv(local_C, local_count, MPI_INT, C, counts, offsets, MPI_INT, 0, MPI_COMM_WORLD);

    int status = EXIT_SUCCESS;
    if (rank == 0) {
        if (save_matrix("A.txt", A, N) && save_matrix("B.txt", B, N) && save_matrix("C.txt", C, N)) {
            printf("Матрицы записаны в A.txt, B.txt и C.txt.\n");
        } else {
            status = EXIT_FAILURE;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(offsets);
    free(counts);
    free(C);
    free(A);
    free(local_A);
    free(B);
    free(local_C);

    MPI_Finalize();
    return status;
}
