import numpy as np; import matplotlib.pyplot as plt;
L=1; T=1.0; J=100; N=100; mu1=1; mu2=0;
tau=T/J; h=L/N;
u=np.zeros((J+1, N+1)); x=np.linspace(0, L, N+1); t=np.linspace(0, T, J+1)
r=1.0; v=tau/h
u[0,:]=0;
for j in range(J):
    u[j+1][N]=mu2
    for i in range(0, N+1): u[j+1][i]=u[j][i-1]
    u[j+1][0]=mu1

# Визуализация решения
plt.figure(figsize=(5, 4))
im = plt.contourf(u)
plt.colorbar(im)
plt.title("Метод характеристик")
plt.show()
