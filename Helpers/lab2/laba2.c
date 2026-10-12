#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define N 324

int main(int argc, char** argv) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A[N][N];  
    double B[N][N];   
    double B_T[N][N];
    double local_A[N][N];  
    double local_B[N][N]; 
    double local_C[N][N];  
    double result[N][N];
    if(rank == 0 )
    {
        int p = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = p;
                B[i][j] = p; 
                p++;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B_T[i][j] = B[j][i];
            }
        } 
    }

    // Раздаем строки матрицы A процессам
    MPI_Scatter(A, N * N / size, MPI_DOUBLE, local_A, N * N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    for (int l = 0; l < N; l++){
        

        MPI_Bcast(B_T[l], N * N / size , MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int j = 0; j < N /size ; j++) {
            local_C[j][l] = 0;
            for (int k = 0; k < N ; k++) {
                
                local_C[j][l] += local_A[j][k] * B_T[l][k];
                //printf("%.0f*%.0f ",local_A[j][k], B_T[l][k]);
            } 
            //printf("%.0f\n", local_C[j][l]);
        }
       
      MPI_Gather(local_C, N*N/size  , MPI_DOUBLE, result, N*N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Результат умножения A * B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f ", result[i][j]);
            }
            printf("\n");
        }

        printf("Время выполнения: %f секунд \n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
