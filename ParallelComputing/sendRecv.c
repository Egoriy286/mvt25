#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Для обмена данными необходимо хотя бы два процесса
    if (world_size < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Ошибка: запустите программу минимум с 2 процессами (-np 2).\n");
        }
        MPI_Finalize();
        return 1;
    }

    int number;

    if (world_rank == 0) {
        // Код выполняется только процессом 0 (отправитель)
        number = 42;
        printf("Процесс 0: отправляю число %d процессу 1...\n", number);

        MPI_Send(
            &number,        // Адрес буфера с данными
            1,              // Количество элементов
            MPI_INT,        // Тип передаваемых данных
            1,              // Ранг процесса-получателя (dest)
            0,              // Тег сообщения (tag)
            MPI_COMM_WORLD  // Коммуникатор
        );
    } else if (world_rank == 1) {
        // Код выполняется только процессом 1 (получатель)
        MPI_Status status;

        MPI_Recv(
            &number,        // Буфер для записи входящих данных
            1,              // Максимальное количество принимаемых элементов
            MPI_INT,        // Ожидаемый тип данных
            0,              // Ранг процесса-отправителя (source)
            0,              // Ожидаемый тег сообщения (tag)
            MPI_COMM_WORLD, // Коммуникатор
            &status         // Структура с метаданными о приеме
        );

        printf("Процесс 1: успешно получил число %d от процесса 0!\n", number);
    }
    

    MPI_Finalize();
    return 0;
}