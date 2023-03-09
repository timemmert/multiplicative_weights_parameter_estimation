import time

import jax
import numpy as np

from constants import dim_state, perturbation_multiplier
from main import main
from util import get_noise

import matplotlib.pyplot as plt

import jax.numpy as jnp

noises = [0.01]
epsilons = [3., 1., 0.1, 0.01]

#T_ends = [100.]
#dts = [0.1]

T_ends = [20.]
dts = [0.02]

#T_ends = [10.]
#dts = [0.01]

#T_ends = [5.]
#dts = [0.005]

#T_ends = [1.]
#dts = [0.001]

horizon_lengths = [0.1]

result_dict = {}

combinations = np.array(np.meshgrid(epsilons, noises, T_ends, dts, horizon_lengths)).T.reshape(-1, 5)
key_measurement = jax.random.PRNGKey(5)
key_perturbation = jax.random.PRNGKey(1)

time_sum = 0
for i, combination in enumerate(combinations):
    print(f"Running combination {i + 1} of {len(combinations)}")
    epsilon, noise, T_end, dt, horizon_length = combination

    ticks_end = int(T_end / dt)
    horizon_length_ticks = int(horizon_length / dt)
    n_horizons = int(np.ceil(T_end / horizon_length))
    key_measurement, subkey_measurement = jax.random.split(key_measurement)
    key_perturbation, subkey_perturbation = jax.random.split(key_perturbation)

    noise_measurement = get_noise(jnp.eye(dim_state) * noise, subkey_measurement, ticks_end=ticks_end)
    perturbation = get_noise(jnp.eye(dim_state) * perturbation_multiplier, subkey_perturbation, ticks_end=ticks_end)

    t_start = time.time()
    p, n = main(epsilon, noise_measurement, perturbation, dt=dt, ticks_end=ticks_end, horizon_length_ticks=horizon_length_ticks, n_horizons=n_horizons)
    time_sum += (time.time() - t_start)
    eta = (len(combinations) - i - 1) * time_sum / (i + 1)
    print(eta)

    result_dict[(noise, epsilon, T_end, dt, horizon_length)] = (p, n)


combinations_without_epsilon = np.array(np.meshgrid(noises, T_ends, dts, horizon_lengths)).T.reshape(-1, 4)
for combination in combinations_without_epsilon:
    noise, T_end, dt, horizon_length = combination
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Probabilities for of each candidate with noise cov={noise}, T={T_end}, dt={dt}, hor={horizon_length}")
    for i, epsilon in enumerate(epsilons):
        p, n = result_dict[(noise, epsilon, T_end, dt, horizon_length)]
        plt.subplot(2, 2, i + 1)
        plt.bar(np.arange(0, n, 1), p)
        plt.title(f"epsilon={epsilon}")

    time_syst = time.time()
    plt.savefig("plots/" + str(time_syst) + ".jpg")
    plt.show()
