import numpy as np
import pytest

from epidemic.model import initialize_space, update_grid


# ---------- initialize_space ----------

def test_initialize_grid_shape_and_dtype():
    g = initialize_space(m=10, n=15, q=10, num_patients_zero=5)
    assert g.shape == (10, 15)
    assert g.dtype == int


def test_initialize_num_infected_matches_requested():
    g = initialize_space(m=20, n=20, q=10, num_patients_zero=50)
    n_infected = int(np.sum(g > 0))
    assert n_infected == 50


def test_initialize_caps_at_total_cells():
    # Запросили больше, чем клеток — должно быть не больше m*n
    g = initialize_space(m=5, n=5, q=10, num_patients_zero=1000)
    assert int(np.sum(g > 0)) == 25


def test_initialize_stages_within_range():
    q = 10
    max_inf = q // 2
    g = initialize_space(m=30, n=30, q=q, num_patients_zero=200)
    nonzero = g[g > 0]
    assert nonzero.min() >= 1
    assert nonzero.max() <= max_inf


# ---------- update_grid ----------

def test_infection_threshold_k1_infects_neighbors():
    # Одна клетка заражена, k=1 — все восприимчивые соседи должны заразиться
    g = np.zeros((3, 3), dtype=int)
    g[1, 1] = 1
    t = np.zeros_like(g)
    g2, _ = update_grid(g, t, q=10, k=1, immunity_duration=0)
    # Центральная клетка прогрессирует, все 8 соседей заражаются → 9 заражённых
    assert int(np.sum(g2 >= 1)) == 9


def test_no_infection_when_below_threshold():
    # Одна заражённая клетка, k=5 — никто не заражается
    g = np.zeros((5, 5), dtype=int)
    g[2, 2] = 1
    t = np.zeros_like(g)
    g2, _ = update_grid(g, t, q=10, k=5, immunity_duration=0)
    # Только центральная клетка остаётся заражённой (прогрессировала)
    assert int(np.sum(g2 > 0)) == 1


def test_last_stage_becomes_recovered():
    q = 10
    max_inf = q // 2  # = 5
    g = np.full((3, 3), max_inf, dtype=int)
    t = np.zeros_like(g)
    g2, _ = update_grid(g, t, q=q, k=1, immunity_duration=0)
    assert (g2 == max_inf + 1).all()


def test_temporary_immunity_expires_to_susceptible():
    # immunity_duration = 1: R-клетка через 1 шаг снова становится S
    q = 10
    max_inf = q // 2
    g = np.full((3, 3), max_inf, dtype=int)
    t = np.zeros_like(g)

    # Шаг 1: все становятся R, таймер = 1
    g1, t1 = update_grid(g, t, q=q, k=1, immunity_duration=1)
    assert (g1 == max_inf + 1).all()
    assert (t1 == 1).all()

    # Шаг 2: таймер истёк → все снова S
    g2, t2 = update_grid(g1, t1, q=q, k=1, immunity_duration=1)
    assert (g2 == 0).all()
    assert (t2 == 0).all()


def test_permanent_immunity_stays_recovered():
    # immunity_duration = 0: R не возвращается в S
    q = 10
    max_inf = q // 2
    g = np.full((3, 3), max_inf + 1, dtype=int)
    t = np.zeros_like(g)
    g2, _ = update_grid(g, t, q=q, k=1, immunity_duration=0)
    assert (g2 >= max_inf + 1).all()