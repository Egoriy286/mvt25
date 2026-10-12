#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
// #define N 4 //840 

int main(int argc, char** argv) {
    
    srand(time(NULL));
    
    int rank, size;
    
    MPI_Init(NULL, NULL);
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
    
    // *a_common 
    double matrix[N][N];
    double vector[N]; // Вектор должен иметь ту же длину, что и количество столбцов в матрице
    int base = N / size;
    int remainder = N % size;

    int rows_per_process = base + (rank < remainder);

    int sendcounts[size], displs[size];
    int recvcounts[size], recvdispls[size];

    int offset = 0;

    for (int p = 0; p < size; p++) {
        int rows = base + (p < remainder);

        sendcounts[p] = rows * N;
        displs[p] = offset * N;

        recvcounts[p] = rows;
        recvdispls[p] = offset;

        offset += rows;
    }

    double local_row[rows_per_process * N]; // Часть матрицы, отправленная процессу
    double local_result[rows_per_process];  // Массив для хранения результата на каждом процессе
    double result[N]; // Массив для сборки результатов

    if(rank == 0 )
    {
        
        // Читаем матрицу
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = rand() % 10 + 1;
            }
        }
        
        // Читаем вектор
        for (int i = 0; i < N; i++) {
            vector[i] = rand()% 10 + 1;
        }
        
        // Выводим матрицу
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f ", matrix[i][j]);
            }
            printf("\n");
        }
        //// Выводим вектор
        printf("\nВектор:\n");
        for (int i = 0; i < N; i++) {
            printf("%.0f ", vector[i]);
        }
        printf("\n");
    }

    

    MPI_Scatterv(
        matrix, sendcounts, displs, MPI_DOUBLE,
        local_row, rows_per_process * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Передаем вектор на все процессы
    MPI_Bcast(vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    
    for (int j = 0; j < rows_per_process; j++) {
        local_result[j] = 0.0;

        for (int k = 0; k < N; k++) {
            local_result[j] += local_row[j * N + k] * vector[k];
        }
    }
    double end_time = MPI_Wtime();
    // Сбор результатов на процессе 0
    MPI_Gatherv(
        local_result, rows_per_process, MPI_DOUBLE,
        result, recvcounts, recvdispls, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        
        printf("Результат умножения матрицы на вектор:\n");
        for (int i = 0; i < N; i++) {
            printf("Результат для строки %d: %10.2f\n", i, result[i]);
        }
        printf("Время выполнения: %f секунд \n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
