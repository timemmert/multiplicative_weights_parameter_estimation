import json
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl

from candidate_response import control_pieces, simulate_controlled_system
from constants import Q_position, T_end, cov_measurement, dim_state, dt, horizon_length, horizon_length_ticks, \
    key_measurement_noise, key_perturbation, linearize_around, measurement_noise_on, \
    n_horizons, noise_multiplier, perturbation, ticks_end, use_ground_truth_control, \
    use_individual_control, \
    use_pieces_control, x_0, x_goal
from losses import loss_maximum_likelihood

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f

mpl.use('TkAgg')

simulation_metadata = {
    "candidate_control_mode": {
        "use_ground_truth_control": use_ground_truth_control,
        "use_individual_control": use_individual_control,
        "use_pieces_control": use_pieces_control,
    },
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

    t_grid = jnp.linspace(
        start=0,
        stop=T_end,
        num=ticks_end
    )

    noise_measurement = jnp.zeros((ticks_end,)) if not measurement_noise_on else jax.random.multivariate_normal(
        key=key_measurement_noise, mean=jnp.zeros((dim_state,)), cov=cov_measurement, shape=(ticks_end + 1,))
    for horizon in range(n_horizons):
        print(horizon)
        # Choose candidate of this horizon step
        j = np.random.choice(
            np.arange(0, n, 1),
            p=p
        )
        K_j, _ = candidates[j]
        #K_j = K_true

        ts = jnp.linspace(
            start=horizon * horizon_length_ticks * dt,
            stop=(horizon + 1) * horizon_length_ticks * dt,
            num=horizon_length_ticks + 1
        )

        n_r = np.zeros((n, horizon_length_ticks + 1, dim_state,))

        # during this horizon, the same model is used
        horizon_offset = horizon * horizon_length_ticks

        x_start = x[horizon_offset]
        x_horizon, u_horizon = simulate_controlled_system(
            K_j,
            f_true,
            ts,
            x_goal,
            x_start,
            perturbation=perturbation[horizon_offset:(horizon + 1) * horizon_length_ticks]
        )
        x[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1] = x_horizon

        x_measure = x_horizon + noise_measurement[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1]

        x_measure_start = x_measure[0]
        for i, candidate in enumerate(candidates):
            _, f_i = candidate

            if use_ground_truth_control:
                x_i = odeint(f_i, x_measure_start, ts, u_horizon)
            elif use_pieces_control:
                x_i = control_pieces(f_i, ts, u_horizon, x_measure)
            elif use_individual_control:
                x_i, _ = simulate_controlled_system(K_j, f_i, ts, x_goal, x_measure_start)
            else:
                raise ValueError()

            n_r[i] = np.asarray(x_measure - x_i)

        # Compute losses, update weights and probabilities
        l = loss_maximum_likelihood(n_r)
        l -= min(l)  # do correction -> debatable!
        l = l / np.sum(l)
        w *= jnp.exp(- epsilon * l)
        p = np.asarray(w / jnp.sum(w))

    plt.plot(x[:, 0], label="MW Controlled system")

    K_ideal, f_ideal = candidates[np.argmax(p)]
    x_ideal, _ = simulate_controlled_system(K_ideal, f_true, t_grid, x_goal, x[0], perturbation=perturbation)
    plt.plot(x_ideal[:, 0], label="Ideal control")

    K_worst, f_worst = candidates[np.argmin(p)]
    x_worst, _ = simulate_controlled_system(K_worst, f_true, t_grid, x_goal, x[0], perturbation=perturbation)
    plt.plot(x_worst[:, 0], label="Worst control")

    plt.legend()
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
