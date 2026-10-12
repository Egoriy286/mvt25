#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>


int main() {
    int num_points = 100000000; // Количество точек
    int num_threads = 8; // Количество потоков
    srand(time(NULL)); // Инициализация генератора случайных чисел
    int total_points_in_circle;
    // Массив для хранения оценок Pi для каждого потока
    double pi_estimates;
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        unsigned int seed = thread_id + 1; // Уникальный seed для каждого потока
        int points_in_circle = 0;

        #pragma omp for reduction(+: total_points_in_circle)
        for (int i = 0; i < num_points; i++) {
            double x = (rand() % 200000) / 100000.0;
            double y = (rand() % 200000) / 100000.0;
            // Проверка, попадает ли точка в круг
            if (x * x + y * y <= 4.0) {
                total_points_in_circle++;
            }
        }

        pi_estimates = 4.0 * (double)total_points_in_circle / (double)num_points;
       
    }

    // Вывод результатов
    double true_pi = M_PI;
    printf("Истинное значение Pi: %.15f\n", true_pi);
    double error = fabs(pi_estimates - true_pi);
    printf("Оценка Pi потоком : %.15f (погрешность: %.15f)\n", pi_estimates, error);
    

    double avg_error = fabs(pi_estimates - true_pi);
    printf("Средняя оценка Pi: %.15f (средняя погрешность: %.15f)\n", pi_estimates, avg_error);

    return 0;
}
