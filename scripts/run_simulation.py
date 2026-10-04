import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from epidemic import initialize_space, update_grid, create_cmap_and_labels

m = 80                    
n = 80                    
q = 20                    
k = 3                     
immunity_duration = 2     
num_patients_zero = 300   
max_steps = 150           

num_simulations = 1
base_seed = 21
csv_filename = "results/epidemic_runs.csv"
gif_filename = "results/epidemic_last_run.gif"


def main():
    
    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "run", "seed",
        "m", "n", "q", "k", "immunity_duration",
        "num_patients_zero", "max_steps",
        "max_I_frac", "extinction_time",
        "final_S_frac", "final_I_frac", "final_R_frac",
    ]

    total_cells = m * n
    max_infected_state = q // 2

    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

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
                im = ax.imshow(grid, cmap=cmap, interpolation="nearest",
                               vmin=0, vmax=q - 1)
                cbar = fig.colorbar(im, ax=ax, ticks=range(q))
                cbar.ax.set_yticklabels(labels)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                def animate(i):
                    nonlocal grid, immunity_timer
                    grid, immunity_timer = update_grid(
                        grid, immunity_timer, q, k, immunity_duration
                    )
                    im.set_array(grid)
                    im.set_clim(0, q - 1)
                    record_current()
                    ax.set_title(
                        f"Эпидемия (шаг {i + 1})\n"
                        f"q={q}, k={k},\n"
                        f"Заражённых: {I_hist[-1]} | Иммунных: {R_hist[-1]}"
                    )
                    return [im]

                ani = animation.FuncAnimation(
                    fig, animate, frames=max_steps, interval=80, blit=True
                )

                try:
                    ani.save(gif_filename, writer="pillow", fps=12)
                    print(f"[прогон {run_idx}] Анимация сохранена в {gif_filename}")
                except Exception as e:
                    print(f"[прогон {run_idx}] Не удалось сохранить анимацию: {e}")

                plt.show()
            else:
                for _ in range(max_steps):
                    grid, immunity_timer = update_grid(
                        grid, immunity_timer, q, k, immunity_duration
                    )
                    record_current()

        
            I_arr = np.array(I_hist, dtype=float)
            max_I_frac = float(I_arr.max() / total_cells) if len(I_arr) else 0.0

            nonzero = np.where(I_arr > 0)[0]
            extinction_time = int(nonzero[-1]) if len(nonzero) else 0

            final_S_frac = S_hist[-1] / total_cells
            final_I_frac = I_hist[-1] / total_cells
            final_R_frac = R_hist[-1] / total_cells

            writer.writerow({
                "run": run_idx,
                "seed": base_seed + run_idx,
                "m": m, "n": n, "q": q, "k": k,
                "immunity_duration": immunity_duration,
                "num_patients_zero": num_patients_zero,
                "max_steps": max_steps,
                "max_I_frac": max_I_frac,
                "extinction_time": extinction_time,
                "final_S_frac": final_S_frac,
                "final_I_frac": final_I_frac,
                "final_R_frac": final_R_frac,
            })

            print(f"[прогон {run_idx}] seed={base_seed + run_idx}, "
                  f"max_I_frac={max_I_frac:.3f}, "
                  f"extinction_time={extinction_time}, "
                  f"final_R_frac={final_R_frac:.3f}")

    print(f"\nСтатистика  в {csv_filename}")


if __name__ == "__main__":
    main()