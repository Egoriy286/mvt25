import numpy as np

def generate_non_singular_matrix_and_vector(size):
    """
    Генерация невырожденной матрицы и вектора.
    :param size: Размерность матрицы.
    :return: Матрица (numpy.ndarray) и вектор (numpy.ndarray).
    """
    while True:
        # Генерация случайной квадратной матрицы
        A = np.random.randint(-10, 10, (size, size)).astype(float)
        # Проверка, что матрица невырожденная (определитель не равен 0)
        if np.linalg.det(A) != 0:
            break

    # Генерация случайного вектора
    b = np.random.randint(-10, 10, size).astype(float)

    return A, b

def save_matrix_and_vector_to_file(filename, A, b):
    """
    Сохранение матрицы и вектора в файл.
    :param filename: Имя файла.
    :param A: Матрица (numpy.ndarray).
    :param b: Вектор (numpy.ndarray).
    """
    with open(filename, 'w') as f:
        # Запись размерности матрицы
        f.write(f"{A.shape[0]}\n")
        # Запись матрицы
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")
        # Запись вектора
        f.write(" ".join(map(str, b)) + "\n")

# Пример использования
size = 4  # Размерность матрицы
A, b = generate_non_singular_matrix_and_vector(size)
x = np.linalg.solve(A, b)  # Решение СЛАУ

# Сохранение в файл
save_matrix_and_vector_to_file("matrix.txt", A, b)

# Сохранение решения в файл

np.savetxt("solution.txt", x, fmt="%.6f")

