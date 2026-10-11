#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ELEMENTS_PER_PROC 2

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_elements = size * ELEMENTS_PER_PROC;
    int *global_array = NULL;
    int *gathered_array = NULL;

    // Память под полные массивы выделяется ТОЛЬКО на процессе root (rank 0)
    if (rank == 0) {
        global_array = (int*)malloc(total_elements * sizeof(int));
        gathered_array = (int*)malloc(total_elements * sizeof(int));

        for (int i = 0; i < total_elements; i++) {
            global_array[i] = (i + 1) * 5; // [5, 10, 15, 20, 25, 30, 35, 40...]
        }

        printf("[Root] Исходный массив: ");
        for (int i = 0; i < total_elements; i++) {
            printf("%d ", global_array[i]);
        }
        printf("\n\n");
    }

    // Локальный буфер выделяется НА КАЖДОМ процессе
    int local_data[ELEMENTS_PER_PROC];

    // 1. РАСПРЕДЕЛЕНИЕ: нарезаем global_array и отправляем по 2 элемента
    MPI_Scatter(
        global_array,       // Отправляемый массив (значим только для root)
        ELEMENTS_PER_PROC,  // Число элементов НА ОДИН процесс!
        MPI_INT,
        local_data,         // Приемный локальный буфер каждого процесса
        ELEMENTS_PER_PROC,  // Сколько принимает каждый процесс
        MPI_INT,
        0,                  // Ранг процесса root
        MPI_COMM_WORLD
    );

    // 2. ЛОКАЛЬНЫЕ ВЫЧИСЛЕНИЯ: каждый процесс меняет свои данные
    for (int i = 0; i < ELEMENTS_PER_PROC; i++) {
        local_data[i] *= 10;
    }

    printf("Процесс %d обработал свои элементы: [%d, %d]\n",
           rank, local_data[0], local_data[1]);

    // 3. СБОРКА: стягиваем измененные данные обратно на процесс 0
    MPI_Gather(
        local_data,         // Отправляемый локальный буфер
        ELEMENTS_PER_PROC,  // Число элементов от каждого процесса
        MPI_INT,
        gathered_array,     // Приемный массив (значим только для root)
        ELEMENTS_PER_PROC,  // Сколько принимаем ОТ КАЖДОГО процесса!
        MPI_INT,
        0,                  // Ранг процесса root
        MPI_COMM_WORLD
    );

    // 4. ВЫВОД ИТОГА
    if (rank == 0) {
        printf("\n[Root] Собранный массив после модификации: ");
        for (int i = 0; i < total_elements; i++) {
            printf("%d ", gathered_array[i]);
        }
        printf("\n");

        free(global_array);
        free(gathered_array);
    }

    MPI_Finalize();
    return 0;
}