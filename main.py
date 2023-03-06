from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f

eps = 0.000003
key_true_model_noise = jax.random.PRNGKey(0)
key_measurement_noise = jax.random.PRNGKey(1)

mpl.use('TkAgg')

dt = 0.005

T_end = 1
ticks_end = int(T_end / dt)

horizon_length = 0.005  # longer horizon -> better performance but also more computational effort. Set to dt for no horizon
horizon_length_ticks = int(horizon_length / dt)

n_horizons = int(np.ceil(T_end / horizon_length))

dim_state = 2
cov_noise_true_model = jnp.eye(dim_state) * 0.1
measurement_noise = False
cov_measurement = jnp.eye(dim_state) * 0.001

use_ground_truth_control = False


xs = jnp.empty((0, dim_state))

x = jnp.array([3.14, 0.])
x_goal = jnp.array([0, 0])
u_j = np.array([0])

g = 9.81
m = 1
l = 1
b = .1


def system_from_parameters(parameters: Tuple, noise: jnp.ndarray = jnp.zeros((ticks_end, dim_state,))):
    A = A_matrix(parameters, x)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters, noise=noise)
    return K, f


n = 8
candidates = []

for l_candidate in np.linspace(0.8, 1.2, n):
    parameters = (g, m, l_candidate, b, dt,)

    K, f = system_from_parameters(parameters=parameters)
    candidates.append((K, f,))


# this noise acts as kind of a simulated model error
noise_true_model = jax.random.multivariate_normal(key=key_true_model_noise, mean=jnp.zeros((dim_state,)), cov=cov_noise_true_model, shape=(ticks_end,))
_, f_true = system_from_parameters(
    parameters=(g, m, l, b, dt),
    noise=noise_true_model
)

p = np.ones((n,)) / n
w = jnp.ones((n,))
epsilon = jnp.sqrt(8 * jnp.log(n) / ticks_end)

noise_measurement = jnp.zeros((ticks_end,)) if not measurement_noise else jax.random.multivariate_normal(key=key_measurement_noise, mean=jnp.zeros((dim_state,)), cov=cov_measurement, shape=(ticks_end,))
u = np.zeros((ticks_end,))
for horizon in range(n_horizons):
    print(horizon)
    # MW algorithm
    j = np.random.choice(
        np.arange(0, n, 1),
        p=p
    )
    K_j, _ = candidates[j]
    # end MW algorithm

    xs = jnp.concatenate(
        (xs, x.reshape(1, -1)),
    )
    ts = jnp.linspace(
        start=horizon * horizon_length_ticks * dt,
        stop=(horizon + 1) * horizon_length_ticks * dt,
        num=horizon_length_ticks + 1
    )

    x_record = np.zeros((horizon_length_ticks + 1, dim_state,))
    n_r = np.zeros((n, horizon_length_ticks + 1, dim_state,))

    # during this horizon, the same model is used, allowing for (likely)
    x_record[0] = x

    for horizon_tick, t in enumerate(ts[:-1]):
        e = x_goal - x
        u[horizon_tick + horizon * horizon_length_ticks] = K_j @ e  # TODO: The same control trajectory will later be used for all systems ???? Or just the matrix?

        # Simulate the real (noisy) system
        x = odeint(
            f_true,
            x,
            jnp.array([t, t+dt]),
            u,
        )[-1, :]
        x_record[horizon_tick + 1] = x + noise_measurement[horizon * horizon_length_ticks + horizon_tick]  # make it noisy

    # in theory, this could be sped up by moving u inside as it is known previously -> later
    for i, candidate in enumerate(candidates):
        _, f_i = candidate

        x_sim_start = x_record[0]
        if use_ground_truth_control:
            x_i = odeint(
                f_i,
                x_sim_start,
                ts,
                u,  # TODO: this here currently takes the control input of the robot, not the individual one -
            )
            n_r[i] = np.asarray(x_record - x_i)
        else:
            x_i = np.zeros((horizon_length_ticks + 1, dim_state,))
            x_i[0] = x_sim_start
            x_sim = x_sim_start
            for horizon_tick, t in enumerate(ts[:-1]):
                e = x_goal - x_sim
                u_individual = K_j @ e  # Note: This still takes the control matrix of the chosen candidate
                x_sim = odeint(
                    f_i,
                    x_sim,
                    jnp.array([t, t+dt]),
                    u_individual,
                )[-1]
                x_i[horizon_tick + 1] = x_sim
            n_r[i] = np.asarray(x_record - x_i)

    def compute_loss_maximum_likelihood(n_r):
        # instead, one might build in a weight matrix cost function, possibly influenced by measurement accuracy
        n_r_reshaped = n_r.reshape(n_r.shape[0], horizon_length_ticks + 1, 1, n_r.shape[2])
        covariances = n_r_reshaped.transpose(0, 1, 3, 2) @ n_r_reshaped
        covariances_sum = np.sum(covariances, axis=1)  # sum along t axis
        return np.log(np.linalg.det(covariances_sum + np.eye(1) * eps))

    def compute_loss_angle_sum(n_r):  # performance a lot worse than MLE cost function
        sum_of_theta_errors_squared = np.sum(n_r[:, :, 0] ** 2, axis=1)
        return sum_of_theta_errors_squared

    l = compute_loss_maximum_likelihood(n_r)
    l -= min(l)  # do correction -> debatable!
    l = l / np.sum(l)
    w *= jnp.exp(- epsilon * l)
    p = np.asarray(w / jnp.sum(w))
    print(sum(p))

print(p)
plt.bar(np.arange(0, n, 1), p)
plt.show()
# plt.plot(xs[:, 0])
# plt.show()
