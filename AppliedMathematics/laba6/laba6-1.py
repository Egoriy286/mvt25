from math import ceil, e, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def initial(x, a):
    return np.exp(-a * x**2)


def characteristics(x, t, a):
    # Ищем начальную координату ξ из ξ + t u₀(ξ) = x.
    x, t = np.broadcast_arrays(x, t)
    left, right = x - t, x
    for _ in range(40):
        mid = (left + right) / 2
        go_right = mid + t * initial(mid, a) < x
        left = np.where(go_right, mid, left)
        right = np.where(go_right, right, mid)
    return initial((left + right) / 2, a)


def samarskii_step(u, tau_over_h, inflow):
    new = np.empty_like(u)
    new[0] = inflow
    for i in range(1, len(u)):
        r = tau_over_h * u[i] / 2
        new[i] = ((1 - r) * u[i] + r * (u[i - 1] + new[i - 1])) / (1 + r)
    return new


def fem_rhs(u, h):
    # Средний поток на каждом элементе P1, интегрированный точно.
    left, right = u[:-1], u[1:]
    flux = (left**2 + left * right + right**2) / 6
    rhs = np.empty_like(u)
    rhs[0] = 0
    rhs[1:-1] = (flux[:-1] - flux[1:]) / h
    rhs[-1] = (flux[-1] - u[-1]**2 / 2) / (h / 2)
    return rhs


def solve(a, length, n_cells, end):
    if not 0 < end < sqrt(e / (2 * a)):
        raise ValueError("Для сравнения с характеристиками нужно 0 < T < t_*.")

    h = length / n_cells
    steps = ceil(end / (0.4 * h))
    x, t = np.linspace(0, length, n_cells + 1), np.linspace(0, end, steps + 1)
    tau = t[1]
    exact = characteristics(x[None, :], t[:, None], a)
    inflow = exact[:, 0]
    inflow_mid = characteristics(0, (t[:-1] + t[1:]) / 2, a)
    fd, fem = np.empty_like(exact), np.empty_like(exact)
    fd[0] = fem[0] = exact[0]

    for j in range(steps):
        fd[j + 1] = samarskii_step(fd[j], tau / h, inflow[j + 1])

        # Три стадии Рунге — Кутты с граничным значением для времени стадии.
        u = fem[j]
        stage1 = u + tau * fem_rhs(u, h)
        stage1[0] = inflow[j + 1]
        stage2 = (3 * u + stage1 + tau * fem_rhs(stage1, h)) / 4
        stage2[0] = inflow_mid[j]
        fem[j + 1] = (u + 2 * (stage2 + tau * fem_rhs(stage2, h))) / 3
        fem[j + 1, 0] = inflow[j + 1]
    return x, t, exact, fd, fem


def plot_results(x, t, exact, fd, fem, a):
    fig, (grid, profile, error) = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    view = min(x[-1], t[-1] + 3 / sqrt(a))
    xi = np.linspace(-t[-1], view, 24)
    grid.plot(xi + t[:, None] * initial(xi, a), t, color="#28798a", lw=1)
    grid.set(title="Лагранжева сетка", xlabel="$x$", ylabel="$t$",
             xlim=(0, view), ylim=(0, t[-1]))

    profile.plot(x, exact[-1], color="0.15", label="Характеристики", lw=2)
    for values, label, color, style in (
        (fd, "МКР (схема А)", "#c55b11", "--"),
        (fem, "МКЭ ($P_1$)", "#17689d", "-."),
    ):
        profile.plot(x, values[-1], color=color, ls=style, label=label, lw=1.8)
        err = np.max(np.abs(values - exact), axis=1)
        error.semilogy(t[1:], err[1:], color=color, ls=style, lw=1.8)

    profile.set(title=rf"Профиль при $t={t[-1]:.3f}$", xlabel="$x$",
                ylabel="$u(x,t)$", xlim=(0, view))
    profile.legend()
    error.set(title="Максимальная ошибка в узлах", xlabel="$t$", ylabel=r"$E_\infty$")
    for ax in (grid, profile, error):
        ax.grid(alpha=0.25)

    out = Path(__file__).resolve().parent / "results" / "laba6-1"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hopf.png", dpi=160)
    if plt.get_backend().lower() != "agg":
        plt.show()


def main():
    a, length, n_cells = 25.0, 2.0, 200
    end = 0.9 * sqrt(e / (2 * a))
    plot_results(*solve(a, length, n_cells, end), a)


if __name__ == "__main__":
    main()
