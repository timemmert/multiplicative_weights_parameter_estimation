from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from numpy import argmax

from candidate_response import candidate_control_function, simulate_control_system_function
from constants import dim_state, linearize_around, minus_goal, plus_goal, print_current_horizon, \
    use_original_sim_start, x_0

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f
from losses import loss_squared_cov


def system_from_parameters(parameters: Tuple):
    # Linearize around resting position
    A = A_matrix(parameters, linearize_around)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters)
    return K, f


def main(
        epsilon: float,
        noise_measurement: jnp.ndarray,
        perturbation: jnp.ndarray,
        dt: float,
        ticks_end: int,
        horizon_length_ticks: int,
        n_horizons: int,
        threshold_stop: Optional[float] = None,
        use_best_covariance: bool = False,  # if False, sample it
        n_l: int = 3,
        n_m: int = 3,
):
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

    n = n_l * n_m
    candidates = []

    for m_candidate in np.linspace(0.8, 1.2, n_m):
        if n_m == 1:
            m_candidate = m
        for l_candidate in np.linspace(0.8, 1.2, n_l):
            parameters = (g, m_candidate, l_candidate, b, dt,)

            K, f = system_from_parameters(parameters=parameters)
            ctrl_fn = candidate_control_function(f)
            candidates.append((K, f, ctrl_fn,))

    K_true, f_true = system_from_parameters(
        parameters=true_parameters,
    )

    simulate_control_system_fn = simulate_control_system_function(f_true)

    p_mult_weights = np.ones((n,)) / n
    global_cov_sum = np.zeros((n, dim_state, dim_state,))
    time_convergence = None
    for horizon in range(n_horizons):
        if print_current_horizon:
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
        x_horizon, u_horizon = simulate_control_system_fn(
            K_true,
            ts,
            x_goal,
            x[horizon_offset],
            dt=dt,
            perturbation=perturbation[horizon_offset:(horizon + 1) * horizon_length_ticks]
        )

        x[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1] = x_horizon
        x_measure = x_horizon + noise_measurement[horizon_offset:((horizon + 1) * horizon_length_ticks) + 1]

        for i, candidate in enumerate(candidates):
            _, f_i, ctrl_fn = candidate
            x_simulation_starts = x_horizon if use_original_sim_start else x_measure
            x_i = ctrl_fn(ts, u_horizon, x_simulation_starts, dt)
            n_r[i] = np.asarray(x_measure - x_i)[1:]

        # Compute losses, update weights and probabilities
        n_r_reshaped = n_r.reshape(n_r.shape[0], n_r.shape[1], 1, n_r.shape[2])
        covariances = n_r_reshaped.transpose(0, 1, 3, 2) @ n_r_reshaped
        global_cov_sum += np.sum(covariances, axis=1)
        cov = global_cov_sum / ((horizon + 1) * horizon_length_ticks)

        if not use_best_covariance:
            j = np.random.choice([i for i in range(n)], p=p_mult_weights)
            cov_sample = cov[j]
        else:
            cov_sample = cov[argmax(p_mult_weights)]

        cov_inv = np.linalg.inv(cov_sample)
        l = loss_squared_cov(n_r, cov_inv)

        l_sum += l
        w = jnp.exp(- epsilon * (l_sum - min(l_sum)))
        p_mult_weights = np.asarray(w / jnp.sum(w))

        if threshold_stop and max(p_mult_weights) > 1 - threshold_stop:
            time_convergence = (horizon + 1) * horizon_length_ticks * dt
            break

    return p_mult_weights, n, time_convergence



