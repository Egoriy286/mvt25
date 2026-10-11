#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 0;

    // Шаг 1: Процесс 0 определяет задачу
    if (rank == 0) {
        N = 1000000LL; // Суммируем числа от 1 до 10 000 000
        printf("[Процесс 0] Общее количество чисел N = %lld\n", N);
        printf("[Процесс 0] Рассылаю N всем процессам через MPI_Bcast...\n");
    }

    // Шаг 2: Рассылка значения N на ВСЕ процессы
    // До этого вызова переменная N на остальных рангах была равна 0
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Шаг 3: Каждый процесс вычисляет свой диапазон
    long long elements_per_proc = N / size;
    long long start = rank * elements_per_proc + 1;
    // Последний процесс забирает возможный остаток от деления
    long long end = (rank == size - 1) ? N : (rank + 1) * elements_per_proc;

    long long local_sum = 0;
    for (long long i = start; i <= end; ++i) {
        local_sum += i;
    }

    printf("Процесс %d: диапазон [%lld ... %lld], локальная сумма = %lld\n", 
           rank, start, end, local_sum);

    // Шаг 4: Сбор и сложение всех локальных сумм на процессе 0
    long long total_sum = 0;
    MPI_Reduce(
        &local_sum,        // Локальное слагаемое каждого процесса
        &total_sum,        // Буфер для глобальной суммы (заполнится на rank 0)
        1,                 // 1 число
        MPI_LONG_LONG,     // Тип данных
        MPI_SUM,           // Операция: сложение
        0,                 // Процесс-получатель (root)
        MPI_COMM_WORLD
    );

    // Шаг 5: Вывод общего итога процессом 0
    if (rank == 0) {
        printf("\n[Процесс 0] Глобальная сумма чисел от 1 до %lld: %lld\n", N, total_sum);
        
        // Аналитическая проверка по формуле Гаусса: N * (N + 1) / 2
        long long expected = N * (N + 1) / 2;
        printf("[Процесс 0] Ожидаемый результат (по формуле): %lld\n", expected);
    }

    MPI_Finalize();
    return 0;
}