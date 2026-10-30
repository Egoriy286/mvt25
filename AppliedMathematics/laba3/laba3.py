# Лабораторная 3. Граничные условия I, II, III рода.
# u_t=((1+x)*u_x)_x-u_x-u+f; x,t ∈ [0,1].
# Модель: u=exp(-t)*(1+x+x²), f=-2*(1+x)*exp(-t).
# Неявная схема, перенос против потока (v=1), прогонка без своих функций.
# Границы с фото: -κ₁λu_x=α₁u+ν₁μ₁; κ₂λu_x=α₂u+ν₂μ₂.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
out = root/'results'; out.mkdir(parents=True,exist_ok=True)
L, T, v, sigma = 1., 1., 1., 1.
meshes = [10,20,40,80]; names = ['Дирихле','Неймана','Робина']
convergence, table, solutions = [], [], []

for kind, name in enumerate(names,1):
    # Одинаковые коэффициенты слева и справа; α=-1 для диссипативного Робина.
    kappa, boundary_alpha, nu = [(0.,-1.,1.),(1.,0.,-1.),(1.,-1.,-1.)][kind-1]
    folder = out/f'type{kind}'; folder.mkdir(exist_ok=True); previous_error = None
    for N in meshes:
        M = N*N; h = L/N; tau = T/M
        x = np.linspace(0,L,N+1); t = np.linspace(0,T,M+1)
        lam_half = 1+(x[:-1]+x[1:])/2
        u = np.zeros((M+1,N+1)); u[0] = 1+x+x*x
        exact = np.exp(-t[:,None])*(1+x+x*x)
        if kind == 1: mu1, mu2 = np.exp(-t), 3*np.exp(-t)
        elif kind == 2: mu1, mu2 = np.exp(-t), -6*np.exp(-t)
        else: mu1, mu2 = np.zeros(M+1), -9*np.exp(-t)

        # Матрица для всех узлов, включая две строки граничных условий.
        lower = np.zeros(N); upper = np.zeros(N); diagonal = np.zeros(N+1)
        lower[:-1] = -tau*(lam_half[:-1]/h**2+v/h)
        upper[1:] = -tau*lam_half[1:]/h**2
        diagonal[1:-1] = 1+tau*((lam_half[:-1]+lam_half[1:])/h**2+v/h+sigma)
        diagonal[0] = kappa*lam_half[0]/h-boundary_alpha
        diagonal[-1] = kappa*lam_half[-1]/h-boundary_alpha
        upper[0] = -kappa*lam_half[0]/h; lower[-1] = -kappa*lam_half[-1]/h
        assert np.all(diagonal > 0) and np.all(lower <= 0) and np.all(upper <= 0)
        assert np.all(diagonal >= np.r_[0,np.abs(lower)]+np.r_[np.abs(upper),0]-1e-12)

        # Коэффициенты прогонки постоянны во времени и вычисляются один раз.
        alpha = np.zeros(N); denominator = np.zeros(N+1)
        denominator[0] = diagonal[0]; alpha[0] = upper[0]/denominator[0]
        for i in range(1,N+1):
            denominator[i] = diagonal[i]-lower[i-1]*alpha[i-1]
            if i < N: alpha[i] = upper[i]/denominator[i]
        assert denominator.min() > 0 and max(abs(alpha)) <= 1+1e-12
        for j in range(M):
            rhs = np.empty(N+1); rhs[1:-1] = u[j,1:-1]-2*tau*(1+x[1:-1])*np.exp(-t[j+1])
            rhs[0] = nu*mu1[j+1]; rhs[-1] = nu*mu2[j+1]
            beta = np.zeros(N+1); beta[0] = rhs[0]/denominator[0]
            for i in range(1,N+1): beta[i] = (rhs[i]-lower[i-1]*beta[i-1])/denominator[i]
            u[j+1,-1] = beta[-1]
            for i in range(N-1,-1,-1): u[j+1,i] = beta[i]-alpha[i]*u[j+1,i+1]

        # Невязки исходных граничных уравнений и всех внутренних строк схемы.
        residual_left = -kappa*lam_half[0]*(u[1:,1]-u[1:,0])/h-boundary_alpha*u[1:,0]-nu*mu1[1:]
        residual_right = kappa*lam_half[-1]*(u[1:,-1]-u[1:,-2])/h-boundary_alpha*u[1:,-1]-nu*mu2[1:]
        rhs_inner = u[:-1,1:-1]-2*tau*(1+x[1:-1])*np.exp(-t[1:,None])
        residual_inner = lower[:-1]*u[1:,:-2]+diagonal[1:-1]*u[1:,1:-1]+upper[1:]*u[1:,2:]-rhs_inner
        boundary_error = max(abs(residual_left).max(),abs(residual_right).max())
        linear_error = abs(residual_inner).max()
        error = abs(u-exact).max(); final_error = abs(u[-1]-exact[-1]).max()
        order = np.nan if previous_error is None else np.log2(previous_error/error)
        assert boundary_error < 1e-9 and linear_error < 1e-9 and np.isfinite(u).all()
        if previous_error is not None: assert error < previous_error
        previous_error = error
        convergence.append([kind,N,M,h,tau,error,final_error,boundary_error,linear_error,order])
        print(f'{name:8s}: N={N:3d}, M={M:4d}, max|u−u*|={error:.6e}, ошибка при t=1: {final_error:.6e}, p={order:.3f}')

        if N == meshes[-1]:
            # Независимое прямое решение на первом и последнем шагах времени.
            matrix = np.diag(diagonal)+np.diag(lower,-1)+np.diag(upper,1)
            for j in [0,M-1]:
                rhs = np.r_[nu*mu1[j+1],u[j,1:-1]-2*tau*(1+x[1:-1])*np.exp(-t[j+1]),nu*mu2[j+1]]
                assert max(abs(np.linalg.solve(matrix,rhs)-u[j+1])) < 1e-9
            np.savez(folder/'solution.npz',x=x,t=t,u=u,exact=exact,mu1=mu1,mu2=mu2,
                     kappa=kappa,alpha=boundary_alpha,nu=nu,L=L,T=T,v=v,sigma=sigma,
                     residual_left=residual_left,residual_right=residual_right)
            np.savetxt(folder/'solution.csv',np.column_stack([x,u[-1],exact[-1],abs(u[-1]-exact[-1])]),
                       delimiter=',',header='x,u_t1,exact_t1,abs_error',comments='')
            for i in [0,N//4,N//2,3*N//4,N]: table.append([kind,x[i],u[-1,i],exact[-1,i],abs(u[-1,i]-exact[-1,i])])
            solutions.append((x,t,u,exact))

convergence = np.array(convergence); table = np.array(table)
np.savetxt(out/'convergence.csv',convergence,delimiter=',',header='type,N,M,h,tau,max_error,final_error,boundary_residual,linear_residual,order',comments='')
np.savetxt(out/'values.csv',table,delimiter=',',header='type,x,u_t1,exact_t1,abs_error',comments='')
model_x, model_t = np.meshgrid(np.linspace(0,L,5),[0.,0.25,0.5,1.])
np.savetxt(out/'model.csv',np.column_stack([model_x.ravel(),model_t.ravel(),(1+model_x).ravel(),
           np.full(model_x.size,v),np.full(model_x.size,sigma),(-2*(1+model_x)*np.exp(-model_t)).ravel(),
           (np.exp(-model_t)*(1+model_x+model_x**2)).ravel()]),delimiter=',',header='x,t,lambda,v,sigma,f,u_exact',comments='')
print('\nЗначения при t=1 (тип, x, численное, точное, ошибка):'); print(table)

fig, ax = plt.subplots(2,3,figsize=(14,8))
for k,(x,t,u,exact) in enumerate(solutions):
    for time in [0.,0.25,0.5,1.]:
        j = round(time/(t[1]-t[0])); ax[0,k].plot(x,u[j],label=f't={time:g}')
    ax[0,k].plot(x,exact[-1],'k--',label='Точное, t=1'); ax[0,k].set(title=names[k],xlabel='x',ylabel='u(x,t)')
    ax[1,k].plot(x,u[-1]-exact[-1]); ax[1,k].set(xlabel='x',ylabel='u−u*, t=1')
    ax[0,k].legend(fontsize=8)
for axis in ax.flat: axis.grid(alpha=0.3)
fig.suptitle('Лабораторная 3: граничные условия; N=80, τ=h²')
fig.tight_layout(); fig.savefig(out/'solution.png',dpi=150)
fig2, axis = plt.subplots(figsize=(7,5))
for kind,name in enumerate(names,1):
    rows = convergence[convergence[:,0] == kind]; axis.loglog(rows[:,3],rows[:,5],'o-',label=name)
axis.loglog(rows[:,3],rows[-1,5]*rows[:,3]/rows[-1,3],'k--',label='O(h)')
axis.set(xlabel='Шаг h (τ=h²)',ylabel='Максимальная ошибка по x и t',title='Сходимость неявной схемы')
axis.legend(); axis.grid(alpha=0.3); fig2.tight_layout(); fig2.savefig(out/'convergence.png',dpi=150)
plt.show()
