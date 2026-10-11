#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

int main(int argc, char** argv) {
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int MAX_LEN = 0;
    if (rank == 0) {
        if (argc == 2) {
            char *end;
            long long length = strtoll(argv[1], &end, 10);
            if (*end == '\0' && length > 0 && length <= INT_MAX) {
                MAX_LEN = (int)length;
            }
        }
    }
    MPI_Bcast(&MAX_LEN, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int counts[size], offsets[size];
    int offset = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = MAX_LEN / size;
        if (i == 0) {
            counts[i] += MAX_LEN % size;
        }
        offsets[i] = offset;
        offset += counts[i];
    }
    int ELEMENTS_PER_PROC = counts[rank];
    int local_data[ELEMENTS_PER_PROC > 0 ? ELEMENTS_PER_PROC : 1];
    int array[MAX_LEN];
    
    if( rank == 0){
        for (int i = 0; i < MAX_LEN; i ++){
            array[i] = rand() % 10;
            printf("%d ", array[i]);
        }
        printf("\n");
    }

    MPI_Scatterv(array, counts, offsets, MPI_INT,
                 local_data, ELEMENTS_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);
    
    int local_sum = 0;
    for (int i = 0; i < ELEMENTS_PER_PROC; i++) {
        local_sum += local_data[i];
        printf("P%d: %d \t", rank, local_data[i]);
    }

    int sums[size];
    MPI_Gather(&local_sum, 1, MPI_INT, 
               sums, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int total_sum = 0;
        for (int i = 0; i < size; i++) {
            total_sum += sums[i];
        }
        printf("\n\t\tSum: %d\n", total_sum);
    }
    MPI_Finalize();
}
