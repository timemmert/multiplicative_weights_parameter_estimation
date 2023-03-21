import json
import time

import jax
import numpy as np

from constants import dim_state, perturbation_multiplier
from main import main
from util import get_noise

import matplotlib.pyplot as plt

import jax.numpy as jnp

result_dict = {}

n_average = 5

epsilon = 1
dt = 0.02
horizon_length = 0.1
t_len = 200
combinations = [
    [epsilon, 0.001, t_len, dt, horizon_length, 3],
    [epsilon, 0.001, t_len, dt, horizon_length, 5],
    [epsilon, 0.001, t_len, dt, horizon_length, 7],
    [epsilon, 0.001, t_len, dt, horizon_length, 9],
    [epsilon, 0.001, t_len, dt, horizon_length, 11],
    [epsilon, 0.001, t_len, dt, horizon_length, 13],
    [epsilon, 0.001, t_len, dt, horizon_length, 15],
    [epsilon, 0.001, t_len, dt, horizon_length, 17],
]

key_measurement = jax.random.PRNGKey(5)
key_perturbation = jax.random.PRNGKey(1)

n_candidates_plot = []
times_convergence_plot = []
times_stddevs_plot = []

use_best_covariance = False # running with False atm
time_sum = 0
n = 0
for i, combination in enumerate(combinations):
    epsilon, noise, T_end, dt, horizon_length, n_l = combination
    print(f"Running combination {i + 1} of {len(combinations)} with n_l {n_l}")

    ticks_end = int(T_end / dt)
    horizon_length_ticks = int(horizon_length / dt)
    n_horizons = int(np.ceil(T_end / horizon_length))

    ts = []
    for i in range(n_average):
        key_measurement, subkey_measurement = jax.random.split(key_measurement)
        key_perturbation, subkey_perturbation = jax.random.split(key_perturbation)

        noise_measurement = get_noise(jnp.eye(dim_state) * noise, subkey_measurement, ticks_end=ticks_end)
        perturbation = get_noise(jnp.eye(dim_state) * perturbation_multiplier, subkey_perturbation, ticks_end=ticks_end)

        p, n, t_conv = main(
            epsilon,
            noise_measurement,
            perturbation,
            dt=dt,
            ticks_end=ticks_end,
            horizon_length_ticks=horizon_length_ticks,
            n_horizons=n_horizons,
            threshold_stop=0.01,
            use_best_covariance=use_best_covariance,
            n_l=n_l,
            n_m=1,
        )

        if t_conv == None:
            t_conv = np.inf
        ts.append(t_conv)
    t_average_random = np.average(ts)
    stddev = np.std(ts)
    times_stddevs_plot.append(stddev)
    n_candidates_plot.append(n_l)
    times_convergence_plot.append(t_average_random)
    print(f"Converged in {t_average_random} seconds on average with stddev {stddev}")

with open(f"data_use-best-cov={use_best_covariance}_multi_candidates.json", "w") as fp:
    results = {
        "noises": n_candidates_plot,
        "times_convergence": times_convergence_plot,
        "times_stddevs": times_stddevs_plot,
    }
    json.dump(results, fp)

plt.errorbar(n_candidates_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o')
plt.show()
