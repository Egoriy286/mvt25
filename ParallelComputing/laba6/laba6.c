#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <errno.h>
#include <fstream> 

// #define N 8        // Размер матрицы
#define MAX_ITER 100000 // Максимальное число итераций
#define TOL 1e-5    // Точность

// Реализация функции fabs
double my_fabs(double x) {
    return (x < 0) ? -x : x;
}


void read_matrix(double** A, double* b, int N) {
    // FILE *file = fopen(filename, "r");
    // if (!file) {
    //     perror("Ошибка открытия файла");
    //     exit(EXIT_FAILURE);
    // }
    srand(time(NULL));
    std::ofstream MatrixFile("A.txt");
    std::ofstream VectorFile("b.txt");
    // Чтение размера матрицы
    // fscanf(file, "%d", &n);
    // int le;
    // Чтение матрицы A
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;

        for (int j = 0; j < N; j++) {
            A[i][j] = (std::rand() % 10) + 1; // Числа от 1 до 10
            row_sum += my_fabs(A[i][j]);
            //fscanf(file, "%lf", &A[i][j]);
        }
        A[i][i] = row_sum + (std::rand() % 10) + 1;
        for (int j = 0; j < N; j++) {
            MatrixFile << A[i][j] << " ";
        }
        MatrixFile << "\n";

    }
    MatrixFile.close();
    // Чтение вектора b
    for (int i = 0; i < N; i++) {
        b[i] = (std::rand() % 100) + 1;
        VectorFile << b[i] << " ";
    }
    VectorFile << "\n";
    VectorFile.close();
    // fclose(file);
}




int main(int argc, char **argv) {
    
    int rank, size;
    int rows_per_process;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    if (rank == 0) {
        if (argc == 2) {
            char *end;
            errno = 0;
            long long length = strtoll(argv[1], &end, 10);
            if (errno == 0 && *end == '\0' && length > 0 && length <= INT_MAX) {
                N = (int)length;
            }
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0 || N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Укажите положительный размер N, кратный числу процессов.\n");
        }
        MPI_Finalize();
        return 1;
    }

    rows_per_process = N / size;
    if (rows_per_process > INT_MAX / N) {
        if (rank == 0) {
            fprintf(stderr, "Слишком большой размер данных для MPI_Scatter.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Строки должны лежать подряд в памяти для MPI_Scatter.
    double **A = (double **)malloc(N * sizeof(double *));
    double *A_data = (double *)malloc((size_t)N * N * sizeof(double));
    if (!A || !A_data) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        A[i] = A_data + (size_t)i * N;
    }
    double *b = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));
    double *x_new = (double *)malloc(N * sizeof(double));
    
    double **local_A = (double **)malloc(rows_per_process * sizeof(double *));
    double *local_A_data = (double *)malloc((size_t)rows_per_process * N * sizeof(double));
    if (!local_A || !local_A_data) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    for (int i = 0; i < rows_per_process; i++) {
        local_A[i] = local_A_data + (size_t)i * N;
    }
    double *local_x_new = (double *)malloc(rows_per_process * sizeof(double));
    double local_norm, global_norm;

    if (!b || !x || !x_new || !local_x_new) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    // Инициализация данных
    if (rank == 0) {
        read_matrix(A, b, N);
        for (int i=0; i<N; i++)
        {
            x[i] = b[i] / A[i][i];
        }
    }

    // Рассылаем части матрицы A строками и вектор b
    MPI_Scatter(A_data, rows_per_process * N, MPI_DOUBLE, local_A_data, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

        for (int i = 0; i < rows_per_process; i++) {
            int global_row = rank * rows_per_process + i;
            local_norm += my_fabs(local_x_new[i] - x[global_row]);
        }

        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                x[i] = x_new[i];
            }
        }

        if (global_norm < TOL) {
            if (rank == 0) {
                printf("Метод Якоби сошелся на %d итерации.\n", iter + 1);
            }
            break;
        }
    }
    double end_time = MPI_Wtime();

    std::ofstream XFile("x.txt");
    // Вывод результатов
    if (rank == 0) {
        printf("Результат:\n");
        for (int i = 0; i < N; i++) {
            printf("x[%d] = %.10f\n", i, x[i]);
            XFile << x[i] << " ";
        }
        XFile << std::endl;
        printf("Время выполнения: %f секунд\n", end_time - start_time);
    }
    XFile.close();
    free(local_x_new);
    free(local_A_data);
    free(local_A);
    free(x_new);
    free(x);
    free(b);
    free(A_data);
    free(A);

    MPI_Finalize();
    return 0;
}
