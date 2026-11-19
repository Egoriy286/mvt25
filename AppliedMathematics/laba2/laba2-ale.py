import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

L, T, N, M, gamma = 1., 0.12, 200, 80, 1.4
gauss = (1 + np.array([-1., 1.]) / np.sqrt(3)) / 2


def masses(x):
    dx = np.diff(x, axis=-1)
    return np.concatenate((dx[..., :1], dx[..., :-1] + dx[..., 1:], dx[..., -1:]), axis=-1) / 2


def fields(q):
    rho, momentum, energy = np.moveaxis(q, -1, 0)
    u = momentum / rho
    p = (gamma - 1) * (energy - momentum * u / 2)
    return rho, u, p


def rhs(x, z, tau):
    # z_i = m_i(x) Q_i. На каждой стадии массы берутся с текущей сетки.
    dx = np.diff(x)
    if dx.min() <= 0:
        raise RuntimeError('Элементы сетки пересеклись')
    q = z / masses(x)[:, None]
    rho, u, p = fields(q)
    if not np.isfinite(q).all() or min(rho.min(), p.min()) <= 0:
        raise RuntimeError('Нефизическое состояние')
    w = u / 2
    cfl = tau * np.max(np.abs(u - w) + np.sqrt(gamma * p / rho)) / dx.min()
    if cfl > 0.5:
        raise RuntimeError(f'Уменьшите шаг по времени: CFL = {cfl:.3f}')

    # Два узла Гаусса на каждом элементе. Производные базиса дают знаки − и +.
    a = gauss[:, None, None]
    qg = (1 - a) * q[:-1] + a * q[1:]
    wg = (1 - gauss[:, None]) * w[:-1] + gauss[:, None] * w[1:]
    _, ug, pg = fields(qg)
    flux = (ug - wg)[..., None] * qg
    flux[..., 1] += pg
    flux[..., 2] += pg * ug
    average = flux.mean(axis=0)
    dz = np.zeros_like(z)
    dz[:-1] -= average
    dz[1:] += average
    dz[[0, -1], 1] = 0
    return w, dz, cfl


def main():
    out = Path(__file__).resolve().parent / 'results' / 'ale'
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, T, M + 1)
    tau = T / M
    X = np.zeros((M + 1, N + 1))
    Z = np.zeros((M + 1, N + 1, 3))
    X[0] = np.linspace(0, L, N + 1)
    p0 = 1 + 0.1 * np.exp(-((X[0] - 0.5) / 0.12)**2)
    q0 = np.column_stack((np.ones_like(p0), np.zeros_like(p0), p0 / (gamma - 1)))
    Z[0] = masses(X[0])[:, None] * q0

    # Один и тот же SSP-RK3 для сетки X и интегралов Z.
    cfl = 0.
    for j in range(M):
        x, z = X[j], Z[j]
        w, dz, c0 = rhs(x, z, tau)
        x1, z1 = x + tau * w, z + tau * dz
        w, dz, c1 = rhs(x1, z1, tau)
        x2 = 0.75 * x + 0.25 * (x1 + tau * w)
        z2 = 0.75 * z + 0.25 * (z1 + tau * dz)
        w, dz, c2 = rhs(x2, z2, tau)
        X[j+1] = (x + 2 * (x2 + tau * w)) / 3
        Z[j+1] = (z + 2 * (z2 + tau * dz)) / 3
        cfl = max(cfl, c0, c1, c2)

    rho, U, P = fields(Z / masses(X)[..., None])
    V, E = 1 / rho, P / ((gamma - 1) * rho)
    mass, _, energy = Z.sum(axis=1).T
    mass_error = np.abs(mass / mass[0] - 1)
    energy_error = np.abs(energy / energy[0] - 1)
    min_dx = np.diff(X, axis=1).min()
    print(f'CFL: {cfl:.3f}, min Δx: {min_dx:.6f}, min ρ: {rho.min():.6f}, min p: {P.min():.6f}')
    print(f'Относительная ошибка массы: {mass_error.max():.2e}, энергии: {energy_error.max():.2e}')
    assert min(min_dx, rho.min(), P.min()) > 0
    assert max(mass_error.max(), energy_error.max()) < 1e-12
    np.savez(out / 'gas.npz', X=X, U=U, V=V, E=E, P=P, t=t, mass=mass,
             mass_error=mass_error, energy=energy, energy_error=energy_error)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    names = ['Давление p', 'Плотность ρ', 'Скорость u']
    times = np.broadcast_to(t[:, None], X.shape)
    for a, f, title in zip(ax.flat, [P, rho, U], names):
        im = a.contourf(X, times, f, levels=20)
        fig.colorbar(im, ax=a)
        a.set(title=title, xlabel='x', ylabel='t')
    ax[1, 1].plot(1e3 * (X[:, ::10] - X[0, ::10]), t, lw=0.8, color='black')
    ax[1, 1].set(title='Смещение узлов ALE', xlabel=r'$10^3 (X_i(t)-X_i(0))$', ylabel='t')
    fig.savefig(out / 'gas.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    lines = []
    for a, f, title in zip(ax, [P, rho, U], names):
        line, = a.plot(X[0], f[0])
        lines.append(line)
        d = 0.05 * np.ptp(f)
        a.set(xlim=(0, L), ylim=(f.min()-d, f.max()+d), title=title, xlabel='x')
        a.grid(alpha=0.3)
    label = fig.suptitle('')

    def animate(j):
        for line, f in zip(lines, [P, rho, U]):
            line.set_data(X[j], f[j])
        label.set_text(f't = {t[j]:.3f}')

    animation = FuncAnimation(fig, animate, frames=range(0, M + 1, 2))
    animation.save(out / 'gas.gif', writer=PillowWriter(fps=15))
    plt.close(fig)


if __name__ == '__main__':
    main()
