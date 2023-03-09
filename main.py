import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from candidate_response import control_pieces, simulate_controlled_system
from constants import dim_state, dt, horizon_length_ticks, linearize_around, \
    loss, \
    minus_goal, n_horizons, perturbation, plus_goal, ticks_end, x_0

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f
from util import get_noise


def system_from_parameters(parameters: Tuple):
    # Linearize around resting position
    A = A_matrix(parameters, linearize_around)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters)
    return K, f


def main(epsilon: float, noise_measurement: jnp.ndarray):
    x = np.zeros((ticks_end + 1, dim_state,))

    x[0] = x_0
    x_goal = plus_goal

    epsilon_goal_reached = 0.1
    # True parameters
    g = 9.81
    m = 1
    l = 1
    l_sum = 0
    b = .1
    true_parameters = (g, m, l, b, dt,)

    n_l = 3
    n_m = 3

    n = n_l * n_m
    candidates = []
    for m_candidate in np.linspace(0.8, 1.2, n_m):
        for l_candidate in np.linspace(0.8, 1.2, n_l):
            parameters = (g, m_candidate, l_candidate, b, dt,)

            K, f = system_from_parameters(parameters=parameters)
            candidates.append((K, f,))

    K_true, f_true = system_from_parameters(
        parameters=true_parameters,
    )

    p = np.ones((n,)) / n
    w = jnp.ones((n,))

    for horizon in range(n_horizons):
        print(horizon)
        ts = jnp.linspace(
            start=horizon * horizon_length_ticks * dt,
            stop=(horizon + 1) * horizon_length_ticks * dt,
            num=horizon_length_ticks + 1
        )
        n_r = np.zeros((n, horizon_length_ticks, dim_state,))

        horizon_offset = horizon * horizon_length_ticks

        if abs(x_goal[0] - x[horizon_offset, 0]) < epsilon_goal_reached:
            if jnp.array_equal(x_goal, plus_goal):
                x_goal = minus_goal
            else:
                x_goal = plus_goal
        x_horizon, u_horizon = simulate_controlled_system(
            K_true,
            f_true,
            ts,
            x_goal,
            x[horizon_offset],
            perturbation=perturbation[horizon_offset:(horizon + 1) * horizon_length_ticks]
        )

        x[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1] = x_horizon
        x_measure = x_horizon + noise_measurement[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1]

        for i, candidate in enumerate(candidates):
            _, f_i = candidate
            x_i = control_pieces(f_i, ts, u_horizon, x_measure)
            n_r[i] = np.asarray(x_measure - x_i)[1:]

        # Compute losses, update weights and probabilities
        l = loss(n_r)
        l = l / np.sum(l)
        l_sum += l
        w = jnp.exp(- epsilon * l_sum)
        p = np.asarray(w / jnp.sum(w))

    # plt.plot(x[:, 0])
    # plt.show()
    return p, n


# fill noisews array with values from 0.000001 to 0.001, multiplying with factor 10
noises = [0.1]
epsilons = [3, 1, 0.1, 0.01]
T_ends = [10]
dts = [0.01]
horizon_lengths = [0.1]

# iterate over all possible combinations of the above array values by creating an array of all possible combinations in a tuple
combinations = np.array(np.meshgrid(noises, epsilons, T_ends, dts, horizon_lengths)).T.reshape(-1, 5)
key = jax.random.PRNGKey(5)

for combination in combinations:
    noise, epsilon, T_end, dt, horizon_length = combination

    ticks_end = int(T_end / dt)
    horizon_length_ticks = int(horizon_length / dt)
    n_horizons = int(np.ceil(T_end / horizon_length))

    p_list = []
    n_list = []
    epsilon_list = []

    key, subkey = jax.random.split(key)
    noise_measurement = get_noise(jnp.eye(dim_state) * noise, subkey)
    p, n = main(epsilon, noise_measurement)

    p_list.append(p)
    n_list.append(n)
    epsilon_list.append(epsilon)


plt.figure(figsize=(10, 10))
plt.suptitle(f"Probabilities for of each candidate with noise cov={noise}, T={T_end}, dt={dt}")
for i in range(len(p_list)):
    plt.subplot(2, 2, i + 1)
    plt.bar(np.arange(0, n_list[i], 1), p_list[i])
    plt.title(f"epsilon={epsilon_list[i]}")

time_syst = time.time()
plt.savefig("plots/" + str(time_syst) + ".jpg")
plt.show()

for T_end in T_ends:
    ticks_end = int(T_end / dt)
    horizon_length_ticks = int(horizon_length / dt)
    n_horizons = int(np.ceil(T_end / horizon_length))

    for noise in noises:
        p_list = []
        n_list = []
        epsilon_list = []

        for epsilon in epsilons:
            key, subkey = jax.random.split(key)
            noise_measurement = get_noise(jnp.eye(dim_state) * noise, subkey)
            p, n = main(epsilon, noise_measurement)
            print(p)
            p_list.append(p)
            n_list.append(n)
            epsilon_list.append(epsilon)
        plt.figure(figsize=(10, 10))
        plt.suptitle(f"Probabilities for of each candidate with noise cov={noise}, T={T_end}, dt={dt}")
        for i in range(len(p_list)):
            plt.subplot(2, 2, i + 1)
            plt.bar(np.arange(0, n_list[i], 1), p_list[i])
            plt.title(f"epsilon={epsilon_list[i]}")

        time_syst = time.time()
        plt.savefig("plots/" + str(time_syst) + ".jpg")
        plt.show()
