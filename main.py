import json
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from candidate_response import control_pieces, simulate_controlled_system
from constants import Q_position, T_end, cov_measurement, dim_state, dt, horizon_length, horizon_length_ticks, \
    key_measurement_noise, linearize_around, loss, measurement_noise_on, \
    minus_goal, n_horizons, noise_multiplier, perturbation, plus_goal, ticks_end, x_0

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f

mpl.use('TkAgg')

simulation_metadata = {
    "measurement": {
        "measurement_covariance": cov_measurement.tolist(),
        "measurement_noise_on": measurement_noise_on,
    },
    "horizon": {
        "horizon_length": horizon_length,
        "horizon_length_ticks": horizon_length_ticks
    },
    "dt": dt,
    "T_end": T_end,
    "ticks_end": ticks_end
}


def system_from_parameters(parameters: Tuple):
    # Linearize around resting position
    A = A_matrix(parameters, linearize_around)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters)
    return K, f


def main():
    x = np.zeros((ticks_end + 1, dim_state,))

    x[0] = x_0
    x_goal = plus_goal

    epsilon_goal_reached = 0.1
    # True parameters
    g = 9.81
    m = 1
    l = 1
    b = .1
    true_parameters = (g, m, l, b, dt,)

    n_l = 5
    n_m = 5

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
    epsilon = jnp.sqrt(8 * jnp.log(n) / ticks_end)

    noise_measurement = jnp.zeros(
        (ticks_end + 1, dim_state,)) if not measurement_noise_on else jax.random.multivariate_normal(
        key=key_measurement_noise, mean=jnp.zeros((dim_state,)), cov=cov_measurement, shape=(ticks_end + 1,))
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
        w *= jnp.exp(- epsilon * l)
        p = np.asarray(w / jnp.sum(w))

    plt.plot(x[:, 0])
    plt.show()
    return p, n


p, n = main()
print(p)
plt.bar(np.arange(0, n, 1), p)
plt.title(f"T={T_end}, dt={dt}, horizon={horizon_length}, meas_noise={noise_multiplier}, Q_position={Q_position}")
time_syst = time.time()
plt.savefig("plots/" + str(time_syst) + ".jpg")
plt.show()
with open("plots/" + str(time_syst) + ".json", "w") as fp:
    json.dump(simulation_metadata, fp)
