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

n_average = 150

dt = 0.02
horizon_length = 0.1
T_end = 200

key_measurement = jax.random.PRNGKey(2)
key_perturbation = jax.random.PRNGKey(2)

use_best_covariance = True  # running with False atm
time_sum = 0
n = 0

epsilon, noise = [10, 0.005]

ticks_end = int(T_end / dt)
horizon_length_ticks = int(horizon_length / dt)
n_horizons = int(np.ceil(T_end / horizon_length))

ts = []
correctness_counter = 0
for i in range(n_average):
    print(i)
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

    correctness_percentage = correctness_counter / (i + 1)
    print(f"Correctness percentage: {correctness_percentage} over {i + 1} runs.")


with open(f"data_use-best-cov={use_best_covariance}_epsilon_high.json", "w") as fp:
    results = {
        "epsilon": epsilon,
        "correctness_percentage": correctness_percentage,
    }
    json.dump(results, fp)
