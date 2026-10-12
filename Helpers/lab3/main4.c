#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512        // Размер матрицы
#define MAX_ITER 100000 // Максимальное число итераций
#define TOL 1e-5    // Точность

// Реализация функции fabs
double my_fabs(double x) {
    return (x < 0) ? -x : x;
}


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




int main(int argc, char **argv) {
    int rank, size;
    int rows_per_process;
    double A[N][N], b[N], x[N], x_new[N];
    double local_A[N][N], local_b[N], local_x_new[N];
    double local_norm, global_norm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Число процессов должно делить размер матрицы N нацело.\n");
        }
        MPI_Finalize();
        return 1;
    }

    rows_per_process = N / size;

    // Инициализация данных
    if (rank == 0) {
        read_matrix("matrix.txt", A, b);
        for (int i=0; i<N; i++)
        {
            x[i] = b[i] / A[i][i];
        }
    }

    // Рассылаем части матрицы A строками и вектор b
    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Начинаем итерации метода Якоби
    double start_time = MPI_Wtime();
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Рассылаем текущие значения x
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вычисляем локальные элементы нового x
        for (int i = 0; i < rows_per_process; i++) {
            int global_row = rank * rows_per_process + i; // Глобальный индекс строки
            double sum = b[global_row];

            for (int j = 0; j < N; j++) {
                if (j != global_row) {
                    sum -= local_A[i][j] * x[j];
                }
            }

            local_x_new[i] = sum / local_A[i][global_row];
        }

        // Собираем локальные части нового x
        MPI_Gather(local_x_new, rows_per_process, MPI_DOUBLE, x_new, rows_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вычисляем Манхэттенскую норму и проверяем критерий сходимости
        local_norm = 0.0;
        if (rank == 0) {
            global_norm = 0.0;
            for (int i = 0; i < N; i++) {
                global_norm += my_fabs(x_new[i] - x[i]);
                x[i] = x_new[i]; // Обновляем x для следующей итерации
            }
        }

        // Рассылаем глобальную норму всем процессам
        MPI_Bcast(&global_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (global_norm < TOL) {
            if (rank == 0) {
                printf("Метод Якоби сошелся на %d итерации.\n", iter + 1);
            }
            break;
        }
    }
    double end_time = MPI_Wtime();

    // Вывод результатов
    if (rank == 0) {
        printf("Результат:\n");
        for (int i = 0; i < N; i++) {
            printf("x[%d] = %.10f\n", i, x[i]);
        }
        printf("Время выполнения: %f секунд\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}