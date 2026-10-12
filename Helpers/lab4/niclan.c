/*1. Реализовать параллельное вычисление числа Пи с помощью рядов\формул. 
   Существует много формул (Википедия поможет). Надо выбрать свой. */
#include <omp.h>
#include <stdio.h>
#include <math.h>

double fabs(double x){
    return (x < 0) ? -x : x; 
}


int main(int argc, char *argv[])
{
    double pi = 0.0;
    #pragma omp parallel for reduction(+:pi) 
    for (int i = 1; i<1000000; i++){
        pi += 1.0 / pow(i, 2);
    }   
    pi = pi*6;
    pi = sqrt(pi);


   
    printf("Приближенное %.32f\n", pi);
    printf("Погрешность  %.32f\n", pi - M_PI);
    return 0;
}
