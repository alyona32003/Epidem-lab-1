import numpy as np
from scipy.ndimage import convolve


def initialize_space(m, n, q, num_patients_zero):
    grid = np.zeros((m, n), dtype=int)
    max_infected = q // 2
    total_cells = m * n
    num_patients_zero = min(num_patients_zero, total_cells)
    flat_indices = np.random.choice(total_cells, size=num_patients_zero, replace=False)
    stages = np.random.randint(1, max_infected + 1, size=num_patients_zero)
    grid.flat[flat_indices] = stages

    return grid


def update_grid(grid, immunity_timer, q, k, immunity_duration, boundary="wrap"):
    """
    immunity_duration 0: постоянный 
    immunity_duration > 0 : временный иммунитет

    """
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