#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 340  // Максимальный размер матрицы

void read_matrix(const char *filename, double A[MAX_SIZE][MAX_SIZE], double b[MAX_SIZE], int *N) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(EXIT_FAILURE);
    }

    // Чтение размера матрицы
    fscanf(file, "%d", N);

    // Чтение матрицы A
    for (int i = 0; i < *N; i++) {
        for (int j = 0; j < *N; j++) {
            fscanf(file, "%lf", &A[i][j]);
            
        }
    }

    // Чтение вектора b
    for (int i = 0; i < *N; i++) {
        fscanf(file, "%lf", &b[i]);
    }

    fclose(file);
}
void check_solution(const char *filename, double x[MAX_SIZE], int N){
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(EXIT_FAILURE);
    }
    double x1[MAX_SIZE];

    // Чтение solution
    for (int i = 0; i < N; i++) {
        fscanf(file, "%lf", &x1[i]);
    }

    fclose(file);

    for (int i = 0; i < N; i++) {
        if (x[i] >= x1[i] - 0.000001 && x[i] <= x1[i] + 0.000001) 
            continue;
        else{
            printf("Решение неверно!\n");
            return;
        }
    }
    printf("Решение верно!\n");
}

double dot(double a[MAX_SIZE], double x[MAX_SIZE], int n, int N){
    int i;
    double sum = 0;
    for (i=(n); i<N; i++){
        sum += a[i] * x[i];
    }
    return sum;
}

// Функция для решения СЛАУ методом Гаусса
void gauss_elimination(double A[MAX_SIZE][MAX_SIZE], double b[MAX_SIZE], double x[MAX_SIZE], int N) {
    int i,j,m; 
    double factor;
    for (m = 0; m < N; m++) {
        for (i = m + 1; i < N; i++) {
            factor = A[i][m] / A[m][m];
            for (j = m; j < N; j++) {
                A[i][j] = A[i][j] - factor * A[m][j];
            }
            b[i] = b[i] - factor * b[m];
            
        }
    }
    for (i = N - 1; i >= 0; i--) {
        double sum = dot(A[i], x, i+1, N);
        x[i] = (b[i] - sum) / A[i][i];
    }
}

int main() {
    // Инициализация матрицы A и вектора b
    double A[MAX_SIZE][MAX_SIZE], b[MAX_SIZE], start, end;
    double x[MAX_SIZE];
    int N; // Размер матрицы
    read_matrix("matrix.txt",A,b, &N);


    // Инициализация вектора x нулям
    for (int i=0; i<N;i++){
        x[i] = 0.0;
    }
    start = omp_get_wtime();

    gauss_elimination(A, b, x, N);

    end = omp_get_wtime();
    check_solution("solution.txt", x, N);

    printf("Время выполнения: %f секунд\n", end - start);
    //printf("[");
    //for (int i = 0; i < N; i++) {
    //    printf("%.3f, \t", x[i]);
    //}
    //printf("]\n");
    return 0;
}
