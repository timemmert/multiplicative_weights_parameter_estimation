from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl

from losses import loss_maximum_likelihood

from dynamical_system import A_matrix, B_matrix, K_matrix, build_f

mpl.use('TkAgg')


key_true_model_noise = jax.random.PRNGKey(0)
key_measurement_noise = jax.random.PRNGKey(1)
dim_state = 2

dt = 0.0005

T_end = 1
ticks_end = int(T_end / dt)

horizon_length = 0.02  # longer horizon -> better performance but also more computational effort. Set to dt for no horizon
horizon_length_ticks = int(horizon_length / dt)

n_horizons = int(np.ceil(T_end / horizon_length))

cov_noise_true_model = jnp.eye(dim_state) * 0.1

measurement_noise_on = True
cov_measurement = jnp.eye(dim_state) * 0.001

# Using ground truth control could be hard to realize in the real world since it needs to be transmitted to the estimator
use_ground_truth_control = False
use_individual_control = False
use_pieces_control = True

assert use_ground_truth_control + use_individual_control + use_pieces_control == 1

loss = loss_maximum_likelihood

xs = jnp.empty((0, dim_state))

x = jnp.array([3.14, 0.])
x_goal = jnp.array([0, 0])
u_j = np.array([0])

# True parameters
g = 9.81
m = 1
l = 1
b = .1
true_parameters = (g, m, l, b, dt,)


def system_from_parameters(parameters: Tuple, noise: jnp.ndarray = jnp.zeros((ticks_end, dim_state,))):
    A = A_matrix(parameters, x)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters, noise=noise)
    return K, f


n_l = 5
n_m = 5

n = n_l
candidates = []

for l_candidate in np.linspace(0.8, 1.2, n_l):
    parameters = (g, m, l_candidate, b, dt,)

    K, f = system_from_parameters(parameters=parameters)
    candidates.append((K, f,))

# this noise acts as kind of a simulated model error
noise_true_model = jax.random.multivariate_normal(key=key_true_model_noise, mean=jnp.zeros((dim_state,)),
                                                  cov=cov_noise_true_model, shape=(ticks_end,))
_, f_true = system_from_parameters(
    parameters=true_parameters,
    noise=noise_true_model
)

p = np.ones((n,)) / n
w = jnp.ones((n,))
epsilon = jnp.sqrt(8 * jnp.log(n) / ticks_end)

noise_measurement = jnp.zeros((ticks_end,)) if not measurement_noise_on else jax.random.multivariate_normal(
    key=key_measurement_noise, mean=jnp.zeros((dim_state,)), cov=cov_measurement, shape=(ticks_end,))
u = np.zeros((ticks_end,))
for horizon in range(n_horizons):
    print(horizon)
    # Choose candidate of this horizon step
    j = np.random.choice(
        np.arange(0, n, 1),
        p=p
    )
    K_j, _ = candidates[j]

    xs = jnp.concatenate(
        (xs, x.reshape(1, -1)),
    )
    ts = jnp.linspace(
        start=horizon * horizon_length_ticks * dt,
        stop=(horizon + 1) * horizon_length_ticks * dt,
        num=horizon_length_ticks + 1
    )

    x_measure = jnp.concatenate(
        (
            jnp.expand_dims(x, axis=0),
            jnp.zeros((horizon_length_ticks, dim_state,))
        )
    )

    n_r = np.zeros((n, horizon_length_ticks + 1, dim_state,))

    # during this horizon, the same model is used
    for horizon_tick, t in enumerate(ts[:-1]):
        global_index = horizon_tick + horizon * horizon_length_ticks
        e = x_goal - x
        u[global_index] = K_j @ e  # TODO: The same control trajectory will later be used for all systems ???? Or just the matrix?

        # Simulate the real (noisy) system
        x = odeint(
            f_true,
            x,
            jnp.array([t, t + dt]),
            u,
        )[-1, :]
        x_measure = x_measure.at[horizon_tick + 1].set(
            x + noise_measurement[horizon * horizon_length_ticks + horizon_tick]
        )

    for i, candidate in enumerate(candidates):
        _, f_i = candidate

        x_sim_start = x_measure[0]
        if use_ground_truth_control:
            x_i = odeint(
                f_i,
                x_sim_start,
                ts,
                u,
            )
        elif use_pieces_control:
            odeint_vec = vmap(lambda x_start_, ts_, u_: odeint(f_i, x_start_, ts_, u_,))
            u_map = jnp.expand_dims(u[horizon*horizon_length_ticks:(horizon+1)*horizon_length_ticks], axis=1)
            ts_expanded = jnp.expand_dims(ts, axis=1)
            t_tuples = jnp.concatenate((ts_expanded, dt + ts_expanded,), axis=1)[:-1]
            x_i = jnp.concatenate(
                (
                    jnp.expand_dims(x_sim_start, axis=0),
                    odeint_vec(x_measure[:-1], t_tuples, u_map)[:, -1, :],
                )
            )
        elif use_individual_control:
            x_i = jnp.concatenate(
                (
                    np.expand_dims(x_sim_start, axis=0),
                    np.zeros((horizon_length_ticks, dim_state,)),
                )
            )
            for horizon_tick, t in enumerate(ts[:-1]):
                e = x_goal - x_i[horizon_tick]
                u_individual = K_j @ e  # Note: This still takes the control matrix of the chosen candidate
                x_i = x_i.at[horizon_tick + 1].set(
                    odeint(
                        f_i,
                        x_i[horizon_tick],
                        jnp.array([t, t + dt]),
                        u_individual,
                    )[-1]
                )
        else:
            raise ValueError()

        n_r[i] = np.asarray(x_measure - x_i)

    # Compute losses, update weights and probabilities
    l = loss_maximum_likelihood(n_r)
    l -= min(l)  # do correction -> debatable!
    l = l / np.sum(l)
    w *= jnp.exp(- epsilon * l)
    p = np.asarray(w / jnp.sum(w))

print(p)
plt.bar(np.arange(0, n, 1), p)
plt.show()
# plt.plot(xs[:, 0])
# plt.show()
