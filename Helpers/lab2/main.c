//Реализовать умножение матрицы на вектор с помощью MPI_Scatter и MPI_Gather. 
//Показать прирост скорости при увеличении количества процессов.
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#define N 100
void read_data(double **matrix, double *vector){
    for (int i = 0; i < N; i++)
    {
        vector[i] = (1+i) * 1.0;
        for (int j = 0; j < N; j++){
            matrix[i][j] = 1.0 / (i + j + 1);
        }
    }
}
void write_data(const char* filename, double result[], int n) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Ошибка открытия файла");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "Результат умножения матрицы на вектор:\n");
    for (int i = 0; i < n; i++) {
        fprintf(file, "Результат для строки %d: %f\n", i, result[i]);
    }

    fclose(file);
}

int main(int argc, char** argv){
    int rank, size;
    double **matrix;
    double *vector;
    double *local_row; // Массив для получения строки матрицы
    double local_result = 0; // Локальный результат умножения
    double *result; // Массив для сборки результатов
    double res, p_res = 0.0;
    MPI_Status status;
    int n, myid, numprocs, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0){
        read_data(matrix, vector);
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                printf("%f ",matrix[i][j]);
            }
            printf("\n");
        }
        for (int i = 0; i < N; i++){
            printf("%f ",vector[i]);
        }
        printf("\n");
    }
    int rows_per_process = N / size; // Сколько строк получит каждый процесс

    MPI_Scatter(&matrix[0][0], N * sizeof(double), MPI_DOUBLE, local_row, N * sizeof(double), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //for (int j = 0; j < rows_per_process; j++) {
    //    for (int k = 0; k < N; k++) {
    //        local_result += local_row[j * N + k] * vector[k];
    //    }
    //}

    for (int j = 0; j < N; j++) {
        local_result += local_row[j] * vector[j];
    }

    if (rank == 0) {
        result = (double *)malloc(size * sizeof(double));
    }
    MPI_Gather(&local_result, 1, MPI_DOUBLE, result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    if (rank == 0) {
        printf("Результат умножения матрицы на вектор:\n");
        for (int i = 0; i < size; i++) {
            printf("Результат для строки %d: %lf\n", i, result[i]);
        }
        write_data("result.txt", result, size);
        printf("Результат записан в файл result.txt\n");

        // Освобождение памяти для результатов
        free(result);
    }
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
    free(local_row);
    MPI_Finalize();
    return 0;
}