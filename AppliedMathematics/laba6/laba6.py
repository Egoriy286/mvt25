# %% [markdown]
# # Лабораторная 6. Метод характеристик
# Обе задачи используют начальный профиль $u_0(x)=e^{-ax^2}$.
# Продолжение гауссианы на всю прямую задаёт
# значения на входе $x=0$. На характеристике из точки $\xi$
# значение $u(x(\xi,t),t)=u_0(\xi)$ остаётся постоянным.

# %%
from math import e, erf, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

a, b = 25.0, 1.0
ell, T = 2.0, 2.0
N, M = 200, 200  # число интервалов по начальной координате и времени

if not np.isfinite([a, b, ell, T]).all() or min(a, ell, T) <= 0 or b < 0:
    raise ValueError("Требуются конечные a, ell, T > 0 и b >= 0.")
if not all(isinstance(n, (int, np.integer)) and n >= 2 for n in (N, M)):
    raise ValueError("N и M должны быть целыми числами не меньше 2.")

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
results = root / "results"
results.mkdir(exist_ok=True)


def initial(x):
    return np.exp(-a * x**2)

# %% [markdown]
# ### Линейный перенос
# Для $u_t+e^{-bt^2}u_x=0$ все характеристики сдвигаются одинаково:
# $x(\xi,t)=\xi+S(t)$, где $S(t)=\int_0^t e^{-bs^2}\,ds$.
# Поэтому $u(x,t)=u_0(x-S(t))$. При $b=0$ получаем $S(t)=t$.

# %%
t_transport = np.linspace(0, T, M + 1)
shift = t_transport
if b > 0:
    scale = sqrt(b)
    shift = sqrt(pi) / (2 * scale) * np.array([erf(scale * t) for t in t_transport])

x0_transport = np.linspace(-shift[-1], ell, N + 1)
x_transport = x0_transport + shift[:, None]

# %% [markdown]
# ### Уравнение Хопфа
# Для $u_t+u\,u_x=0$ скорость зависит от начальной точки:
# $x(\xi,t)=\xi+t\,u_0(\xi)$.
# При $t_*=-1/\min u_0'=\sqrt{e/(2a)}$ профиль впервые получает
# вертикальную касательную. Строим решение до $\min(T,t_*)$ включительно.

# %%
t_break = sqrt(e / (2 * a))
t_hopf = np.linspace(0, min(T, t_break), M + 1)
x0_hopf = np.linspace(-t_hopf[-1], ell, N + 1)  # скорость не превышает 1
u0_hopf = initial(x0_hopf)
x_hopf = x0_hopf + t_hopf[:, None] * u0_hopf

# %% [markdown]
# ### Графики
# На карте цвет соответствует $u(x,t)$, белые линии отмечают характеристики.

# %%
def plot_solution(x, t, u0, title, filename):
    # x[j, i] — положение i-й характеристики в момент t[j].
    n_times, n_paths = x.shape
    fig, (profiles, spacetime) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    fig.suptitle(title)
    for j in np.linspace(0, n_times - 1, min(5, n_times), dtype=int):
        profiles.plot(x[j], u0, label=rf"$t={t[j]:.3g}$")
    profiles.set(xlabel="$x$", ylabel="$u(x,t)$", title="Профили решения",
                 xlim=(0, ell), ylim=(0, 1.05))
    profiles.grid(alpha=0.3)
    profiles.legend()

    times = np.broadcast_to(t[:, None], x.shape)
    values = np.broadcast_to(u0, x.shape)
    image = spacetime.contourf(x, times, values, levels=np.linspace(0, 1, 21),
                               cmap="viridis")
    lines = np.linspace(0, n_paths - 1, min(19, n_paths), dtype=int)
    spacetime.plot(x[:, lines], t, color="white", lw=0.7, alpha=0.65)
    spacetime.set(xlabel="$x$", ylabel="$t$", title="Решение и характеристики",
                  xlim=(0, ell), ylim=(0, t[-1]))
    fig.colorbar(image, ax=spacetime, label="$u(x,t)$", ticks=np.linspace(0, 1, 6))
    fig.savefig(results / filename, dpi=150)

plot_solution(
    x_transport, t_transport, initial(x0_transport),
    r"Линейный перенос  $u_t+e^{-bt^2}u_x=0$", "transport.png",
)
plot_solution(x_hopf, t_hopf, u0_hopf, r"Уравнение Хопфа  $u_t+u\,u_x=0$", "hopf.png")

if plt.get_backend().lower() != "agg":
    plt.show()
