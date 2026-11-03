
# ## Теплопроводность — компактная версия
# $\rho c T_t=(\lambda T_x)_x$: пенопласт — кирпич — пенопласт.
# Слева 20 °C, справа и в начале внутри −50 °C. Учебные свойства слоёв согласованы в чате.
# Неявная схема, прогонка внутри цикла; своих функций нет.

import numpy as np; import matplotlib.pyplot as plt
from pathlib import Path
out = Path('results/compressed'); out.mkdir(parents=True,exist_ok=True)
d = np.array([0.05,0.25,0.05]); lam = np.array([0.04,0.7,0.04]); rho = np.array([30.,1800.,30.]); c = np.array([1400.,840.,1400.])
T_left, T_right, T0 = 20., -50., -50.; N = 20; T = 7*24*3600; M = 1008; tau = T/M
borders = np.r_[0,np.cumsum(d)]
x = np.r_[np.linspace(borders[0],borders[1],N+1),np.linspace(borders[1],borders[2],N+1)[1:],np.linspace(borders[2],borders[3],N+1)[1:]]
h = np.diff(x); G = np.repeat(lam,N)/h; capacity = np.repeat(rho*c,N)*h; C = (capacity[:-1]+capacity[1:])/2
t = np.linspace(0,T,M+1); u = np.full((M+1,3*N+1),T0); u[:,0] = T_left; u[:,-1] = T_right

a = -tau*G[1:-1]; b = C+tau*(G[:-1]+G[1:]); n = 3*N-1
q1 = np.zeros(M); q2 = np.zeros(M); balance = np.zeros(M)
for j in range(M):
    rhs = C*u[j,1:-1]; rhs[0] += tau*G[0]*T_left; rhs[-1] += tau*G[-1]*T_right
    alpha = np.zeros(n-1); beta = np.zeros(n); alpha[0] = a[0]/b[0]; beta[0] = rhs[0]/b[0]
    for i in range(1,n):
        z = b[i]-a[i-1]*alpha[i-1]
        if i < n-1: alpha[i] = a[i]/z
        beta[i] = (rhs[i]-a[i-1]*beta[i-1])/z
    u[j+1,-2] = beta[-1]
    for i in range(n-2,-1,-1): u[j+1,i+1] = beta[i]-alpha[i]*u[j+1,i+2]
    q1[j] = G[0]*(u[j+1,0]-u[j+1,1]); q2[j] = G[-1]*(u[j+1,-2]-u[j+1,-1])
    dE = np.sum(C*(u[j+1,1:-1]-u[j,1:-1])); heat = tau*(q1[j]-q2[j]); balance[j] = abs(dE-heat)/max(abs(dE),abs(heat),1.)

q_stationary = (T_left-T_right)/np.sum(d/lam); resistance = np.r_[0,np.cumsum(1/G)]; u_stationary = T_left-q_stationary*resistance
A = np.diag(G[:-1]+G[1:])+np.diag(-G[1:-1],1)+np.diag(-G[1:-1],-1)
rhs = np.zeros(n); rhs[0] = G[0]*T_left; rhs[-1] = G[-1]*T_right; u_check = np.linalg.solve(A,rhs)
print('Стационарный поток:',q_stationary,'Вт/м²; температуры стыков:',u_stationary[N],u_stationary[2*N])
print('Ошибка стационарного решения:',max(abs(u_check-u_stationary[1:-1])),'; баланса:',balance.max())
assert balance.max() < 1e-7 and max(abs(u_check-u_stationary[1:-1])) < 1e-8
assert u.min() >= T_right-1e-8 and u.max() <= T_left+1e-8
np.savez(out/'heat.npz',x=x,t=t,u=u,q1=q1,q2=q2,balance=balance,stationary=u_stationary)
np.savetxt(out/'temperature.csv',np.column_stack([x,u[0],u[6],u[36],u[144],u[-1],u_stationary]),delimiter=',',header='x,t0,t1h,t6h,t24h,t168h,stationary',comments='')
np.savetxt(out/'flux.csv',np.column_stack([t[1:],q1,q2]),delimiter=',',header='time_s,q_left,q_right',comments='')


fig, ax = plt.subplots(1,2,figsize=(12,4))
for j in [0,6,36,144,M]: ax[0].plot(x,u[j],label=f't = {t[j]/3600:g} ч')
ax[0].plot(x,u_stationary,'k--',label='Стационарное'); ax[0].set(xlabel='x, м',ylabel='Температура, °C')
ax[1].plot(t[1:]/3600,q1,label='Слева'); ax[1].plot(t[1:]/3600,q2,label='Справа')
ax[1].axhline(q_stationary,color='black',linestyle='--',label='Стационарный поток'); ax[1].set(xlabel='Время, ч',ylabel='Поток по +x, Вт/м²')
for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out/'heat.png',dpi=150); plt.show()

# ### Вывод
# Стенка прогревается слева; стационарный поток 24.5 Вт/м², стыки −10.625 и −19.375 °C.
# Через 7 суток решение ещё приближается к стационарному. Результаты — в `results/compressed/`.
