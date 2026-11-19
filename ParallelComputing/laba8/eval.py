import numpy as np

A, B, C = (np.loadtxt(name, ndmin=2) for name in ('A.txt', 'B.txt', 'C.txt'))
if not A.size or A.shape[0] != A.shape[1] or A.shape != B.shape or A.shape != C.shape:
    raise SystemExit('Ожидаются три квадратные матрицы одинакового размера.')

error = np.max(np.abs(A @ B - C))
if error != 0:
    raise SystemExit(f'Матрица C содержит ошибку. Максимальная разность равна {error:g}.')
print('Проверка пройдена. Максимальная разность |A @ B - C| = 0.')
