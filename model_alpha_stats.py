import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
from matplotlib.colors import ListedColormap
import csv

def initialize_space(m, n, q, num_patients_zero):
    grid = np.zeros((m, n), dtype=int)
    max_infected = q // 2
    total_cells = m * n
    num_patients_zero = min(num_patients_zero, total_cells)
    flat_indices = np.random.choice(total_cells, size=num_patients_zero, replace=False)
    stages = np.random.randint(1, max_infected + 1, size=num_patients_zero)
    grid.flat[flat_indices] = stages

    return grid


def update_grid(grid, immunity_timer, q, k, immunity_duration, boundary='wrap'):
    m, n = grid.shape
    max_infected = q // 2
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    infected_mask = (grid >= 1) & (grid <= max_infected)
    infected_neighbors = convolve(infected_mask.astype(int), kernel,
                                  mode=boundary, cval=0)
    new_grid = grid.copy()
    new_timer = immunity_timer.copy()

    susceptible = (grid == 0)
    new_infections = susceptible & (infected_neighbors >= k)
    new_grid[new_infections] = 1


    mildly_sick = (grid >= 1) & (grid < max_infected)
    new_grid[mildly_sick] = grid[mildly_sick] + 1

  
    becoming_recovered = (grid == max_infected)
    new_grid[becoming_recovered] = max_infected + 1

    if immunity_duration == 0:
        progressing_R = (grid > max_infected) & (grid < q - 1)
        new_grid[progressing_R] = grid[progressing_R] + 1
    else:
        new_timer[becoming_recovered] = immunity_duration

        already_R = (grid > max_infected)
        new_timer[already_R] = immunity_timer[already_R] - 1

        expiring = already_R & (immunity_timer == 1)
        new_grid[expiring] = 0
        new_timer[expiring] = 0

        still_immune = already_R & (immunity_timer > 1)
        new_grid[still_immune] = max_infected + 1

    return new_grid, new_timer


def create_cmap_and_labels(q):
    max_infected = q // 2
    num_recovered = q - 1 - max_infected
    colors = ['#1f77b4']
    labels = ['S (0)']

    for i in range(1, max_infected + 1):
        ratio = (i - 1) / max(1, max_infected - 1) if max_infected > 1 else 0
        colors.append((1.0, 0.8 - 0.6 * ratio, 0.0))
        labels.append(f'I_{i} ({i})')

    for i in range(max_infected + 1, q):
        ratio = (i - max_infected - 1) / max(1, num_recovered - 1) if num_recovered > 1 else 0
        colors.append((0.2, 0.9 - 0.5 * ratio, 0.2))
        labels.append(f'R_{i - max_infected} ({i})')

    return ListedColormap(colors), labels


m = 80                    
n = 80                    
q = 20                     
k = 3                     
immunity_duration = 2     
num_patients_zero = 300   
max_steps = 150           


num_simulations = 1                     
base_seed = 21                           # Базовое зерно ГСЧ
csv_filename = "epidemic_runs.csv"       
gif_filename = "epidemic_last_run.gif"   
fieldnames = [
    'run', 'seed',
    'm', 'n', 'q', 'k', 'immunity_duration',
    'num_patients_zero', 'max_steps',
    'max_I_frac', 'extinction_time',
    'final_S_frac', 'final_I_frac', 'final_R_frac',
]
csv_file = open(csv_filename, "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

total_cells = m * n
max_infected_state = q // 2

for run_idx in range(num_simulations):
    np.random.seed(base_seed + run_idx)

    grid = initialize_space(m, n, q, num_patients_zero)
    immunity_timer = np.zeros_like(grid, dtype=int)

    S_hist, I_hist, R_hist = [], [], []

    def record_current():
        S_hist.append(int(np.sum(grid == 0)))
        I_hist.append(int(np.sum((grid >= 1) & (grid <= max_infected_state))))
        R_hist.append(int(np.sum(grid > max_infected_state)))

    record_current()  

    is_last = (run_idx == num_simulations - 1)

    if is_last:
      
        cmap, labels = create_cmap_and_labels(q)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=q - 1)
        cbar = fig.colorbar(im, ax=ax, ticks=range(q))
        cbar.ax.set_yticklabels(labels)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def animate(i):
            global grid, immunity_timer
            grid, immunity_timer = update_grid(grid, immunity_timer, q, k, immunity_duration)
            im.set_array(grid)
            im.set_clim(0, q - 1)
            record_current()
            n_infected = I_hist[-1]
            n_recovered = R_hist[-1]
            ax.set_title(
                f'Эпидемия (шаг {i + 1})\n'
                f'q={q}, k={k},\n'
                f'Заражённых: {n_infected} | Иммунных: {n_recovered}'
            )
            return [im]

        ani = animation.FuncAnimation(fig, animate, frames=max_steps,
                                      interval=80, blit=True)

    
        try:
            ani.save(gif_filename, writer='pillow', fps=12)
            print(f"[прогон {run_idx}] Анимация сохранена в {gif_filename}")
        except Exception as e:
            print(f"[прогон {run_idx}] Не удалось сохранить анимацию: {e}")

        plt.show()
    else:
       
        for _ in range(max_steps):
            grid, immunity_timer = update_grid(grid, immunity_timer,
                                               q, k, immunity_duration)
            record_current()

    
    I_arr = np.array(I_hist, dtype=float)
    max_I_frac = float(I_arr.max() / total_cells) if len(I_arr) else 0.0

    nonzero = np.where(I_arr > 0)[0]
    extinction_time = int(nonzero[-1]) if len(nonzero) else 0

    final_S_frac = S_hist[-1] / total_cells
    final_I_frac = I_hist[-1] / total_cells
    final_R_frac = R_hist[-1] / total_cells

    writer.writerow({
        'run': run_idx,
        'seed': base_seed + run_idx,
        'm': m, 'n': n, 'q': q, 'k': k,
        'immunity_duration': immunity_duration,
        'num_patients_zero': num_patients_zero,
        'max_steps': max_steps,
        'max_I_frac': max_I_frac,
        'extinction_time': extinction_time,
        'final_S_frac': final_S_frac,
        'final_I_frac': final_I_frac,
        'final_R_frac': final_R_frac,
    })

    print(f"[прогон {run_idx}] seed={base_seed + run_idx}, "
          f"max_I_frac={max_I_frac:.3f}, "
          f"extinction_time={extinction_time}, "
          f"final_R_frac={final_R_frac:.3f}")

csv_file.close()
print(f"\nСтатистика в {csv_filename}")