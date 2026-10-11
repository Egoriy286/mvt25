#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    // printf("Hello world from processor %s, rank %d out of %d processors\n",
    // processor_name, world_rank, world_size);
    int MAX_MSG_LEN = 100;
    char message[MAX_MSG_LEN];
    if (world_rank == 0){
        snprintf(message, sizeof(message), "Hello world");
        MPI_Send(&message, MAX_MSG_LEN, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    if (world_rank != 0){
        MPI_Recv(&message, MAX_MSG_LEN, MPI_CHAR, world_rank-1, 0, MPI_COMM_WORLD, &status);
        printf("Процесс %d: успешно получил сообщение: %s от процесса %d!\n", world_rank, message, world_rank-1);
        if (world_rank < world_size -1){
        MPI_Send(&message, MAX_MSG_LEN, MPI_CHAR, world_rank+1, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}