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
        #if np.linalg.det(A) != 0:
        #    break
        # Диагональные элементы большие симметричные
        np.fill_diagonal(A, np.diag(A) + size)
        if np.all(np.linalg.eigvals(A) > 0):
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
size = 340  # Размерность матрицы
A, b = generate_non_singular_matrix_and_vector(size)

# Сохранение в файл
save_matrix_and_vector_to_file("matrix.txt", A, b)

# Сохранение решения в файл
x = np.linalg.solve(A, b)
np.savetxt("solution.txt", x, fmt="%.6f")



#def Gauss(A:np.ndarray, b:np.ndarray):
#    n = b.shape[0]  # Размерность систе
#    # Прямой ход метода Гаусса
#    #print(A)
#    factor = 0.0
#    for m in range(n):
#        for i in range(m + 1, n):
#            factor = A[i][m] / A[m][m]
#            for j in range(m, n):
#                A[i][j] -= factor * A[m][j]
#                #print(A[i][j])
#            b[i] -= factor * b[m]
#    # Обратный ход
#    x = np.zeros(n)
#    for i in range(n - 1, -1, -1):
#        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]
#        print(A[i, i + 1:], x[i + 1:])
#        print(np.dot(A[i, i + 1:], x[i + 1:]))
#
#    return x
#x = Gauss(A, b)
#print(x)
