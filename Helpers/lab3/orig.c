#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#define N 512
void read_matrix(const char *filename, double A[N][N], double b[N]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(EXIT_FAILURE);
    }
    int n;
    // Чтение размера матрицы
    fscanf(file, "%d", &n);

    // Чтение матрицы A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file, "%lf", &A[i][j]);
        }
    }

    // Чтение вектора b
    for (int i = 0; i < N; i++) {
        fscanf(file, "%lf", &b[i]);
    }

    fclose(file);
}

int main(){
    double A[N][N], b[N];
    double x[N]; // Вектор решений
    read_matrix("matrix.txt", A, b);
    for (int i=0; i<N; i++)
        x[i] = b[i] / A[i][i];
    double local = 0;
    double start_time = MPI_Wtime();
    for (int k=0; k<1000000; k++){
        for (int i=0; i<N; i++)
        {
            local = b[i];
            for (int j = 0; j < N; j++)
            {
                if(i!=j)
                {
                    local = local - A[i][j]*x[j];
                }
            }
            local = local / A[i][i];
            x[i] = local;
        }
    }
    double end_time = MPI_Wtime();
    for(int i=0; i<N;i++)
    {
        printf("x[%d]=%.20f\n", i,x[i]);
    }
    printf("Время выполнения: %f секунд\n", end_time - start_time);
    return 0;
}