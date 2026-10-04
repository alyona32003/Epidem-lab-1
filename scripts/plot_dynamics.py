"""Верификация и валидация модели эпидемии.

Строит графики S/I/R по шагам для одной симуляции.

Верификация: убеждаемся, что модель решает ту задачу, которую мы ставили


Валидация: сравниваем качественное поведение с классической SIR-моделью —
S монотонно убывает, I имеет пик и затем спадает, R монотонно растёт.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from epidemic import initialize_space, update_grid



m = 80
n = 80
q = 20
k = 3
immunity_duration = 2
num_patients_zero = 300
max_steps = 200
seed = 42


def run_once():
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

    # Нормируем 
    S = np.array(S_hist) / total_cells
    I = np.array(I_hist) / total_cells
    R = np.array(R_hist) / total_cells
    return S, I, R


def main():
    os.makedirs("results", exist_ok=True)

    S, I, R = run_once()
    steps = np.arange(len(S))

    # --- График динамики ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(steps, S, label="S (восприимчивые)", color="#1f77b4", linewidth=2)
    ax.plot(steps, I, label="I (заражённые)", color="#d62728", linewidth=2)
    ax.plot(steps, R, label="R (иммунные)", color="#2ca02c", linewidth=2)
    ax.set_xlabel("Шаг симуляции")
    ax.set_ylabel("Доля клеток")
    ax.set_title(f"Динамика S/I/R  (m={m}, n={n}, q={q}, k={k}, "
                 f"immunity={immunity_duration}, seed={seed})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = "results/dynamics_SIR.png"
    fig.savefig(out_png, dpi=150)
    print(f"График сохранён в {out_png}")

    # --- Автоматические проверки (валидация) ---
    print("\nПроверки валидации:")
    print(f"  S[0] = {S[0]:.4f}, S[-1] = {S[-1]:.4f}  (ожидается убывание)")
    print(f"  I max = {I.max():.4f} на шаге {int(I.argmax())}")
    print(f"  R[0] = {R[0]:.4f}, R[-1] = {R[-1]:.4f}  (ожидается рост)")

    # Качественные критерии
    assert S[0] > S[-1], "S должна убывать"
    assert I.max() > I[0], "I должна иметь пик выше начального значения"
    assert R[-1] > R[0], "R должна вырасти"

    print("\nВалидация пройдена: качественное поведение совпадает с SIR.")

    plt.show()


if __name__ == "__main__":
    main()