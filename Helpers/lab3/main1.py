import numpy as np

def construct_matrix(N):
    r = -1
    A = np.zeros((N-1, N-1))

    for i in range(N-1):
        A[i, i] = - 2 * r
        if i > 0:
            A[i, i-1] = r
        if i < N-2:
            A[i, i+1] = r
    return A

def save(filename, A, b):
    """
    Сохранение матрицы и вектора в файл.
    :param filename: Имя файла.
    :param A: Матрица (numpy.ndarray).
    :param b: Вектор (numpy.ndarray).
    :param x: Вектор (numpy.ndarray).
    """
    with open(filename, 'w') as f:
        # Запись размерности матрицы
        f.write(f"{A.shape[0]}\n")
        # Запись матрицы
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")
        # Запись вектора
        f.write(" ".join(map(str, b)) + "\n")
        
l = 1.0       
N = 512        
h = l / N

A = construct_matrix(N+1)
print(A.shape[0])

print("Матрица A:")
print(A)

x = np.linspace(0, l, N)

b = np.zeros(N) 


b[0] = 1        # ГУ_1 = 1
b[-1] = 0       # ГУ_2 = 0
print(b)
save("matrix.txt", A, b)
x = np.linalg.solve(A, b)
for i in range(len(x)):
    print(f" {x[i]:.10f}")