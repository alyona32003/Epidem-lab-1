import numpy as np
import click

from .model import initialize_space, update_grid


@click.command()
@click.option("--m", default=80, show_default=True, type=int,
              help="Ширина решётки")
@click.option("--n", default=80, show_default=True, type=int,
              help="Высота решётки")
@click.option("--q", default=20, show_default=True, type=int,
              help="Всего состояний (q > 2)")
@click.option("--k", default=3, show_default=True, type=int,
              help="Порог заражения (k >= 1)")
@click.option("--immunity", "immunity_duration", default=2, show_default=True,
              type=int,
              help="Длительность иммунитета в шагах (0 = постоянный)")
@click.option("--patients0", "num_patients_zero", default=300, show_default=True,
              type=int,
              help="Начальное число заражённых")
@click.option("--steps", "max_steps", default=150, show_default=True, type=int,
              help="Количество шагов симуляции")
@click.option("--seed", default=21, show_default=True, type=int,
              help="Зерно генератора случайных чисел")
def main(m, n, q, k, immunity_duration, num_patients_zero, max_steps, seed):
    
    np.random.seed(seed)

    grid = initialize_space(m, n, q, num_patients_zero)
    immunity_timer = np.zeros_like(grid, dtype=int)

    max_infected_state = q // 2
    total_cells = m * n

    S_hist, I_hist, R_hist = [], [], []

    def record():
        S_hist.append(int(np.sum(grid == 0)))
        I_hist.append(int(np.sum((grid >= 1) & (grid <= max_infected_state))))
        R_hist.append(int(np.sum(grid > max_infected_state)))

    record()

    for _ in range(max_steps):
        grid, immunity_timer = update_grid(
            grid, immunity_timer, q, k, immunity_duration
        )
        record()

    I_arr = np.array(I_hist, dtype=float)
    max_I_frac = float(I_arr.max() / total_cells) if len(I_arr) else 0.0

    nonzero = np.where(I_arr > 0)[0]
    extinction_time = int(nonzero[-1]) if len(nonzero) else 0

    final_S = S_hist[-1] / total_cells
    final_I = I_hist[-1] / total_cells
    final_R = R_hist[-1] / total_cells

    click.echo(f"Параметры: m={m}, n={n}, q={q}, k={k}, "
               f"immunity={immunity_duration}, patients0={num_patients_zero}, "
               f"steps={max_steps}, seed={seed}")
    click.echo(f"Пиковая доля заражённых max_I_frac = {max_I_frac:.4f}")
    click.echo(f"Время до исчезновения инфекции extinction_time = {extinction_time}")
    click.echo(f"Финальные доли: S={final_S:.4f}, I={final_I:.4f}, R={final_R:.4f}")


if __name__ == "__main__":
    main()