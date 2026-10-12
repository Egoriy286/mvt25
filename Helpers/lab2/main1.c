#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define N 4 //840 

int main(int argc, char** argv) {
    
    srand(time(NULL));
    
    int rank, size;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // *a_common 
    double matrix[N][N];
    double vector[N]; // Вектор должен иметь ту же длину, что и количество столбцов в матрице
    int rows_per_process = N / size;

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
        /*
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
        printf("\n");*/
    }

    

    MPI_Scatter(matrix, rows_per_process * N, MPI_DOUBLE, local_row, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Передаем вектор на все процессы
    MPI_Bcast(vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    
    for (int j = 0; j < rows_per_process; j++) {
        local_result[j] = vector[j];  // Инициализация для каждой строки
        for (int k = 0; k < N; k++) {
            if(rank != j)
                local_result[j] -= local_row[j * N + k] ;
        }
        if(rank == j)
            local_result[j] /= local_row[j * N];
    }
    double end_time = MPI_Wtime();
    // Сбор результатов на процессе 0
    MPI_Gather(local_result, rows_per_process, MPI_DOUBLE, result, rows_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    

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
