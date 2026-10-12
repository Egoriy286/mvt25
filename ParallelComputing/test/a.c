//Реализовать умножение матрицы на вектор с помощью MPI_Scatter и MPI_Gather. 
//Показать прирост скорости при увеличении количества процессов.
#include <stdio.h>
#include <mpi.h>
#include <math.h>
void read_vectors(double *x, double *y){
    for (int i = 0; i < 100; i++)
    {
        x[i] = i * 1.0;
        y[i] = i * 2.0;
    }
}

int main(int argc, char** argv){
    int rank, size;
    double x[100], y[100];
    double res, p_res = 0.0;
    MPI_Status status;
    int n, myid, numprocs, i, N;
    N = 100;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    /* Initialize x and y arrays */
    if (0 == myid)
        read_vectors(x, y);

    MPI_Bcast(x, N, MPI_DOUBLE, 0, comm);
    MPI_Bcast(y, N, MPI_DOUBLE, 0, comm);

    for (i = myid * N/numprocs; i < (myid + 1) * N/numprocs; i++)
    {
        p_res = p_res + x[i] * y[i];
    }
    if (0 == myid){
        for ( i=1; i<numprocs; i++)
            MPI_Recv(&res, 1, MPI_DOUBLE, i, 0, comm, &status);
        p_res = p_res + res;
    }
    else 
        MPI_Send(&p_res, 1, MPI_DOUBLE, 0, 0, comm);
    if(0 == myid)
        printf("Result: %10.2f\n", p_res);
    MPI_Finalize();
    return 0;
}