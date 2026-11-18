import numpy as np

a = np.loadtxt("A.txt")
b = np.loadtxt("b.txt")

print(np.linalg.solve(a, b))
