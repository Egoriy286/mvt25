# Теплопроводность: кирпич — пенопласт; пенопласт только справа.
# Слева T=20 °C; справа Робин: −λ T_x=h_wind*(T_surface−T_air).
# Воздух и начальная температура −50 °C; ветер 5 м/с — учебный сценарий.
# McAdams: h_wind=5.7+3.8*v; EnergyPlus Engineering Reference, McAdams:
# https://bigladdersoftware.com/epx/docs/24-2/engineering-reference/outside-surface-heat-balance.html#mcadams
# Пенопласт 5 см: λ=0.04, ρ=30, c=1400; кирпич 25 см: λ=0.7, ρ=1800, c=840.
# Учебные свойства постоянны; пенофола нет.
# Неявная схема, прогонка внутри цикла; своих функций нет.

import numpy as np; import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__file__).resolve().parent/'results/laba4-6'; out.mkdir(parents=True,exist_ok=True)
d = np.array([0.25,0.05]); lam = np.array([0.7,0.04])
rho = np.array([1800.,30.]); c = np.array([840.,1400.])
T_left, T_air, T0 = 20., -50., -50.; v = 5.; h_wind = 5.7+3.8*v
N = 20; T = 7*24*3600; M = 1008; tau = T/M
borders = np.r_[0,np.cumsum(d)]; K = len(d)*N
x = np.r_[0,np.concatenate([np.linspace(borders[k],borders[k+1],N+1)[1:] for k in range(len(d))])]
h = np.diff(x); G = np.repeat(lam,N)/h; capacity = np.repeat(rho*c,N)*h; C = (capacity[:-1]+capacity[1:])/2
t = np.linspace(0,T,M+1); u = np.full((M+1,K+1),T0); u[:,0] = T_left

# Граничный узел без теплоёмкости: G[-1]*(T[-2]−T[-1])=h_wind*(T[-1]−T_air).
# Исключаем T[-1]: последний отрезок и конвекция — последовательные сопротивления.
G_right = 1/(1/G[-1]+1/h_wind)
a = -tau*G[1:-1]; b = C+tau*(G[:-1]+G[1:]); b[-1] = C[-1]+tau*(G[-2]+G_right); n = K-1
q1 = np.zeros(M); q2 = np.zeros(M); balance = np.zeros(M)
for j in range(M):
    rhs = C*u[j,1:-1]; rhs[0] += tau*G[0]*T_left; rhs[-1] += tau*G_right*T_air
    alpha = np.zeros(n-1); beta = np.zeros(n); alpha[0] = a[0]/b[0]; beta[0] = rhs[0]/b[0]
    for i in range(1,n):
        z = b[i]-a[i-1]*alpha[i-1]
        if i < n-1: alpha[i] = a[i]/z
        beta[i] = (rhs[i]-a[i-1]*beta[i-1])/z
    u[j+1,-2] = beta[-1]
    for i in range(n-2,-1,-1): u[j+1,i+1] = beta[i]-alpha[i]*u[j+1,i+2]
    u[j+1,-1] = (G[-1]*u[j+1,-2]+h_wind*T_air)/(G[-1]+h_wind)
    q1[j] = G[0]*(T_left-u[j+1,1]); q2[j] = h_wind*(u[j+1,-1]-T_air)
    dE = np.sum(C*(u[j+1,1:-1]-u[j,1:-1])); heat = tau*(q1[j]-q2[j]); balance[j] = abs(dE-heat)/max(abs(dE),abs(heat),1.)

q_stationary = (T_left-T_air)/(np.sum(d/lam)+1/h_wind); resistance = np.r_[0,np.cumsum(1/G)]; u_stationary = T_left-q_stationary*resistance
# Независимая стационарная система включает наружную поверхность и строку Робина.
diagonal = np.r_[G[:-1]+G[1:],G[-1]+h_wind]
A = np.diag(diagonal)+np.diag(-G[1:],1)+np.diag(-G[1:],-1)
rhs = np.zeros(K); rhs[0] = G[0]*T_left; rhs[-1] = h_wind*T_air; u_check = np.linalg.solve(A,rhs)
error = max(abs(u_check-u_stationary[1:])); robin_error = max(abs(G[-1]*(u[1:,-2]-u[1:,-1])-q2))
print(f'Ветер: {v:g} м/с; h_wind={h_wind:g} Вт/(м²·К); стационарный поток: {q_stationary:.6f} Вт/м²')
print('Стыки, °C:',u_stationary[np.arange(1,len(d))*N]); print('Наружная поверхность в стационаре, °C:',u_stationary[-1])
print('Ошибка стационара:',error,'; баланса:',balance.max(),'; Робина, Вт/м²:',robin_error)
print('Отклонение от стационара через 7 суток, °C:',max(abs(u[-1]-u_stationary)))
assert balance.max() < 1e-7 and error < 1e-8 and robin_error < 1e-8
assert u.min() >= T_air-1e-8 and u.max() <= T_left+1e-8
np.savez(out/'heat.npz',x=x,t=t,u=u,q1=q1,q2=q2,balance=balance,stationary=u_stationary,d=d,lam=lam,rho=rho,c=c,v=v,h_wind=h_wind)
np.savetxt(out/'temperature.csv',np.column_stack([x,u[0],u[6],u[36],u[144],u[-1],u_stationary]),delimiter=',',header='x,t0,t1h,t6h,t24h,t168h,stationary',comments='')
np.savetxt(out/'flux.csv',np.column_stack([t[1:],q1,q2]),delimiter=',',header='time_s,q_left,q_right',comments='')

fig, ax = plt.subplots(1,2,figsize=(12,4))
for j in [0,6,36,144,M]: ax[0].plot(x,u[j],label=f't = {t[j]/3600:g} ч')
ax[0].plot(x,u_stationary,'k--',label='Стационарное'); ax[0].set(xlabel='x, м',ylabel='Температура, °C')
for border in borders[1:-1]: ax[0].axvline(border,color='gray',linestyle=':',alpha=0.5)
ax[1].plot(t[1:]/3600,q1,label='Слева'); ax[1].plot(t[1:]/3600,q2,label='Справа (Робин)')
ax[1].axhline(q_stationary,color='black',linestyle='--',label='Стационарный поток'); ax[1].set(xlabel='Время, ч',ylabel='Поток по +x, Вт/м²')
for axis in ax: axis.legend(fontsize=8); axis.grid(alpha=0.3)
fig.suptitle(f'Пенопласт только справа; Робин справа, ветер {v:g} м/с')
plt.tight_layout(); plt.savefig(out/'heat.png',dpi=150); plt.show()
