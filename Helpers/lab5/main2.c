#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 1000  // Максимальный размер матрицы

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

// Функция для решения СЛАУ методом Гаусса
void gauss_elimination(double A[MAX_SIZE][MAX_SIZE], double b[MAX_SIZE], double x[MAX_SIZE], int N) {
    int i,j,k; 
    double M[MAX_SIZE][MAX_SIZE];

    for (i = 0; i < N; i++){
        for (j = 0; j < N+1; j++) {
            M[i][j] = A[i][j];    
        }
    }

    // Прямой ход (Зануление нижнего левого угла)
    for ( k = 0; k < N; k++) //k-номер строки
    {
        for ( i = 0; i < N + 1; i++) //i-номер столбца
            M[k][i] = M[k][i] / A[k][k]; //Деление k-строки на первый член !=0 для преобразования его в единицу
        for ( i = k + 1; i < N; i++) //i-номер следующей строки после k
        {
            double K = M[i][k] / M[k][k]; //Коэффициент
            for ( j = 0; j < N + 1; j++) //j-номер столбца следующей строки после k
                M[i][j] = M[i][j] - M[k][j] * K; //Зануление элементов матрицы ниже первого члена, преобразованного в единицу
        }
        for ( i = 0; i < N; i++) //Обновление, внесение изменений в начальную матрицу
            for ( j = 0; j < N + 1; j++)
                A[i][j] = M[i][j];
    }


    // Обратный ход (Зануление верхнего правого угла)
    for ( k = N - 1; k > -1; k--) //k-номер строки
    {
        for ( i = N; i > -1; i--) //i-номер столбца
            M[k][i] = M[k][i] / A[k][k];
        for ( i = k - 1; i > -1; i--) //i-номер следующей строки после k
        {
            double K = M[i][k] / M[k][k];
            for ( j = N; j > -1; j--) //j-номер столбца следующей строки после k
                M[i][j] = M[i][j] - M[k][j] * K;
        }
    }
    

    for (int i = 0; i < N; i++)
        x[i] = M[i][N];
}

int main() {
    // Инициализация матрицы A и вектора b
    double A[MAX_SIZE][MAX_SIZE], b[MAX_SIZE], start, end;
    double x[MAX_SIZE];
    int N; // Размер матрицы
    read_matrix("matrix.txt",A,b, &N);
    for (int i = 0; i < N; i++)
        A[i][N] = b[i];

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
    //printf("]\N");
    return 0;
}
