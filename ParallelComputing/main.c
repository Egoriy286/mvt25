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
    int number;
    if (world_rank == 0){
        number = 42;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    if (world_rank != 0){
        MPI_Recv(&number, 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &status);
        printf("Процесс %d: успешно получил  %d от процесса 0!\n", world_rank, number);
        if (world_rank < world_size -1){
        MPI_Send(&number, 1, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}