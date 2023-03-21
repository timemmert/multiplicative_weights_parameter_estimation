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

n_average = 10

epsilon = 1
dt = 0.02
horizon_length = 0.1
t_len = 200
combinations = [
    [0.1, 0.005, t_len, dt, horizon_length],
    [0.5, 0.005, t_len, dt, horizon_length],
    [1, 0.005, t_len, dt, horizon_length],
    [5, 0.005, t_len, dt, horizon_length],
    [10, 0.005, t_len, dt, horizon_length],
    #[50, 0.005, t_len, dt, horizon_length],
]

key_measurement = jax.random.PRNGKey(5)
key_perturbation = jax.random.PRNGKey(1)

epsilons_plot = []
times_convergence_plot = []
times_stddevs_plot = []
correctness_percentage = []


use_best_covariance = True # running with False atm
time_sum = 0
n = 0
for i, combination in enumerate(combinations):
    epsilon, noise, T_end, dt, horizon_length = combination
    print(f"Running combination {i + 1} of {len(combinations)} with epsilon {epsilon}")

    ticks_end = int(T_end / dt)
    horizon_length_ticks = int(horizon_length / dt)
    n_horizons = int(np.ceil(T_end / horizon_length))

    ts = []
    correctness_counter = 0
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
        )
        if np.argmax(p) == int((n - 1)/2):
            correctness_counter += 1

        if t_conv == None:
            t_conv = np.inf
        ts.append(t_conv)
    t_average_random = np.average(ts)
    stddev = np.std(ts)
    times_stddevs_plot.append(stddev)
    epsilons_plot.append(epsilon)
    correctness_percentage.append(correctness_counter / n_average)
    times_convergence_plot.append(t_average_random)
    print(f"Converged in {t_average_random} seconds on average with stddev {stddev}")

with open(f"data_use-best-cov={use_best_covariance}_epsilon.json", "w") as fp:
    results = {
        "epsilons": epsilons_plot,
        "times_convergence": times_convergence_plot,
        "times_stddevs": times_stddevs_plot,
        "correctness_percentage": correctness_percentage,
    }
    json.dump(results, fp)

plt.errorbar(epsilons_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o')
plt.show()
