#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
double f(double x){
    return x*x + 5*x + x * 3.14;
}
int main(int argc, char **argv) {
    double result=0;
    int rank, size;
    

    double local_sum = 0;
    int n = 1000;   
    double a = 0;   
    double b = 10; 
    double h =  ((b - a) / n);  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    for (int i = rank * n / size; i < (rank+1) * n / size ; i++){
        
        double x = a + i * h;
        local_sum +=  h*(f(x) + f(x+h))/2;
    }
    MPI_Reduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        printf("%f \n",result);
    }

    MPI_Finalize();
    return 0;
}