#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char** argv) {
    srand(time(NULL));
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 0;
    if (rank == 0) {
        if (argc == 2) {
            char *end;
            long long length = strtoll(argv[1], &end, 10);
            if (*end == '\0' && length > 0) {
                N = (int)length;
            }
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (N <= 0) {
        MPI_Finalize();
        return 0;
    }

    int base_rows = N / size;
    
    double* matrix = NULL;
    double* vector = (double*)malloc(N * sizeof(double));
    double* result = NULL;

    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        result = (double*)malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            vector[i] = rand() % 10 + 1;
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = rand() % 10 + 1;
            }
        }
        
        printf("Матрица (%d x %d):\n", N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f ", matrix[i * N + j]);
            }
            printf("\n");
        }
        
        printf("\nВектор:\n");
        for (int i = 0; i < N; i++) {
            printf("%.0f ", vector[i]);
        }
        printf("\n\n");
    }

    double* local_row = (double*)malloc(base_rows * N * sizeof(double));
    double* local_result = (double*)malloc(base_rows * sizeof(double));

    MPI_Scatter(matrix, base_rows * N, MPI_DOUBLE,
                local_row, base_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int j = 0; j < base_rows; j++) {
        local_result[j] = 0.0;
        for (int k = 0; k < N; k++) {
            local_result[j] += local_row[j * N + k] * vector[k];
        }
    }

    if (rank == 0) {
        int remainder = N % size;
        int processed_rows = base_rows * size;
        for (int i = 0; i < remainder; i++) {
            int current_row = processed_rows + i;
            result[current_row] = 0.0;
            for (int k = 0; k < N; k++) {
                result[current_row] += matrix[current_row * N + k] * vector[k];
            }
        }
    }

    double end_time = MPI_Wtime();

    MPI_Gather(local_result, base_rows, MPI_DOUBLE,
               result, base_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Результат умножения матрицы на вектор:\n");
        for (int i = 0; i < N; i++) {
            printf("Результат для строки %d: %10.2f\n", i, result[i]);
        }
        printf("Время выполнения: %f секунд \n", end_time - start_time);
        free(matrix);
        free(result);
    }

    free(vector);
    free(local_row);
    free(local_result);

    MPI_Finalize();
    return 0;
}
