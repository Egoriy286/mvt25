from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
JOULES_PER_KWH = 3_600_000


def snapshot_steps(t):
    """Вернуть ближайшие шаги к контрольным моментам и последний шаг."""
    steps = {len(t) - 1}
    for seconds in (3600, 6 * 3600, 24 * 3600):
        if seconds <= t[-1]:
            steps.add(int(np.argmin(abs(t[1:] - seconds))) + 1)
    return sorted(steps)


def write_summary(
    out, *, layers, d, N, t, u, q_left, q_right, capacity, T0, T_out,
    u_stationary, q_stationary, stationary_error, balance_error,
    h_wind=None, wind_speed=None, robin_error=None,
):
    wall = ' — '.join(f'{name} {100 * width:g} см' for name, width in zip(layers, d))
    if h_wind is None:
        right_boundary = f'справа задана температура поверхности {T_out:g} °C'
    else:
        right_boundary = (
            f'справа условие Робина `q = hᵥ(Tпов − Tвозд)` при температуре воздуха {T_out:g} °C, '
            f'ветре {wind_speed:g} м/с и `hᵥ = {h_wind:g} Вт/(м²·К)`'
        )

    # В неявной схеме поток за шаг берётся в конце этого шага.
    dt = np.diff(t)
    energy_in = (dt @ q_left) / JOULES_PER_KWH
    energy_out = np.cumsum(dt * q_right) / JOULES_PER_KWH
    energy_stored = (capacity @ (u[-1, 1:-1] - u[0, 1:-1])) / JOULES_PER_KWH
    final_deviation = np.max(abs(u[-1] - u_stationary))

    joints = np.arange(1, len(d)) * N
    columns = ['Время, ч', *(f'T стыка {i}, °C' for i in range(1, len(d))),
               'T справа, °C', 'q слева, Вт/м²', 'q справа, Вт/м²', 'Ушло наружу, кВт·ч/м²']
    rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---:'] * len(columns)) + ' |']
    for step in snapshot_steps(t):
        values = [*u[step, joints], u[step, -1], q_left[step - 1], q_right[step - 1]]
        cells = [f'{t[step] / 3600:g}', *(f'{value:.6f}' for value in values), f'{energy_out[step - 1]:.6g}']
        rows.append('| ' + ' | '.join(cells) + ' |')

    checks = [
        f'- Макс. нормированная невязка баланса за шаг {balance_error:.3e} при допуске 1e-7.',
        f'- Макс. расхождение сеточного и аналитического стационарных профилей '
        f'{stationary_error:.3e} °C при допуске 1e-8 °C.',
        f'- Температура на всей сетке от {u.min():g} до {u.max():g} °C в пределах граничных '
        f'и начальной температур с допуском 1e-8 °C.',
    ]
    if robin_error is not None:
        checks.append(f'- Макс. невязка условия Робина {robin_error:.3e} Вт/м² при допуске 1e-8 Вт/м².')

    summary = [
        '# Теплопроводность стенки', '',
        '## Постановка', '',
        f'Одномерная теплопроводность `ρc ∂T/∂t = ∂/∂x(λ ∂T/∂x)` в стенке «{wall}». '
        f'Слева {u[0, 0]:g} °C, внутри в начале {T0:g} °C, {right_boundary}. '
        'На стыках непрерывны температура и тепловой поток.', '',
        f'Неявная схема с прогонкой. {N} отрезков на слой, шаг {dt[0]:g} с, '
        f'длительность {t[-1] / 3600:g} ч.', '',
        '## Температуры, потоки и потери энергии', '', *rows, '',
        'Последний столбец показывает, сколько тепла ушло наружу от начала расчёта '
        'до указанного часа, по сумме `q справа ⋅ Δt`.', '',
        f'Стационарные теплопотери {q_stationary:.6f} Вт/м². К концу расчёта максимальное '
        f'отклонение температуры от стационарного профиля {final_deviation:.6g} °C.', '',
        f'За {t[-1] / 3600:g} ч на 1 м² стенки подано слева **{energy_in:.3f} кВт·ч**, '
        f'ушло наружу **{energy_out[-1]:.3f} кВт·ч**, осталось в стенке **{energy_stored:.3f} кВт·ч**.', '',
        '## Проверки', '', *checks, '',
        '[График температур и потоков](heat.png)', '',
    ]
    (out / 'summary.md').write_text('\n'.join(summary), encoding='utf-8')
