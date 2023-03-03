from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
from dynamical_system import A_matrix, B_matrix, K_matrix, build_f

mpl.use('TkAgg')

T_end = 10
dt = 0.1
Ticks_end = int(T_end / dt)
xs = jnp.empty((0, 2))

x = jnp.array([3.14, 0.])
x_goal = jnp.array([0, 0])
u_j = np.array([0])

g = 9.81
m = 1
l = 1
b = .1


def system_from_parameters(parameters: Tuple):
    A = A_matrix(parameters, x)
    B = B_matrix(parameters)

    K = K_matrix(A_jax=A, B_jax=B)
    f = build_f(parameters)
    return K, f

n = 30
candidates = []

for l_candidate in np.linspace(0.8, 1.2, n):
    parameters = (g, m, l_candidate, b,)

    K, f = system_from_parameters(parameters=parameters)
    candidates.append((K, f,))

_, f_true = system_from_parameters(parameters=(g, m, l, b,))

p = np.ones((n,)) / n
w = jnp.ones((n,))
epsilon = jnp.sqrt(8 * jnp.log(n) / Ticks_end)

for ticks in range(Ticks_end):
    print(ticks)
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
    t = jnp.array([ticks * dt, (ticks + 1) * dt])

    x_old = x
    e = x_goal - x_old
    u_j = K_j @ e

    x = odeint(
        f_true,
        x,
        t,
        u_j
    )[-1, :]

    n_r = np.empty((n, 2,))
    for i, candidate in enumerate(candidates):
        _, f_i = candidate

        x_i = odeint(
            f_i,
            x_old,
            t,
            u_j
        )[-1, :]

        n_r[i] = np.asarray(x - x_i)
    l = n_r[:, 0] ** 2  # TODO: Improve this loss computation
    l = l / np.sum(l)
    w *= jnp.exp(- epsilon * l)
    p = np.asarray(w / jnp.sum(w))
    print(sum(p))

print(p)
plt.bar(np.arange(0, n, 1), p)
plt.show()
# plt.plot(xs[:, 0])
# plt.show()
