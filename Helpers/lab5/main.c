#include <omp.h>
#include <stdio.h>
#include <string.h>

#define PASSWORD "zzaz" // Укажите пароль
#define PASSWORD_LENGTH 4


int main() {
    double start, end;
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    const int charset_size = strlen(charset);
    char found_password[PASSWORD_LENGTH + 1] = {0};

    int found = 0; // Переменная для отслеживания, найден ли пароль
    omp_set_num_threads(2);
    // Начало измерения времени
    start = omp_get_wtime();

    // Перебор всех возможных комбинаций
    #pragma omp parallel for collapse(4) shared(found, found_password) 
    for (int i = 0; i < charset_size; i++) {
        for (int j = 0; j < charset_size; j++) {
            for (int k = 0; k < charset_size; k++) {
                for (int l = 0; l < charset_size; l++) {
                    // Если пароль уже найден, пропускаем выполнение
                    if (found) continue;

                    // Проверка на совпадение с паролем
                    if (charset[i] == PASSWORD[0] && charset[j] == PASSWORD[1] &&
                        charset[k] == PASSWORD[2] && charset[l] == PASSWORD[3]) {
                        
                        // Критическая секция для безопасного обновления переменной found и found_password
                        #pragma omp atomic write
                        found = 1;

                        #pragma omp critical
                        {
                            found_password[0] = charset[i];
                            found_password[1] = charset[j];
                            found_password[2] = charset[k];
                            found_password[3] = charset[l];
                            found_password[4] = '\0';
                        }
                    }
                }
            }
        }
    }

    // Конец измерения времени
    end = omp_get_wtime();

    if (found) {
        printf("Пароль найден: %s\n", found_password);
    } else {
        printf("Пароль не найден.\n");
    }

    printf("Время выполнения: %f секунд\n", end - start);

    return 0;
}
