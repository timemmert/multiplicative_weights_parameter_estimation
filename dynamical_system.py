from typing import Tuple

import jax.numpy as jnp

# damped single pendulum: http://underactuated.mit.edu/pend.html
import numpy as np
from control import lqr


def A_matrix(parameters: Tuple, x_lin: jnp.ndarray):
    g, m, l, b, dt = parameters
    return jnp.array([
        [0, 1],
        [-g / l * jnp.cos(x_lin[0]), -b / (m * l ** 2)]
    ])


def B_matrix(parameters: Tuple):
    g, m, l, b, dt = parameters
    return jnp.array(
        [
            [0],
            [1 / (m * l ** 2)]
        ]
    )


def K_matrix(A_jax: jnp.ndarray, B_jax: jnp.ndarray):
    Q = np.zeros((2, 2,))
    Q[0, 0] = 1
    Q[1, 1] = 1
    R = np.array([[1]])
    A = np.asarray(A_jax)
    B = np.asarray(B_jax)
    K, _, _ = lqr(A, B, Q, R)
    return jnp.asarray(K)


def build_f(parameters: Tuple, noise: jnp.ndarray):
    g, m, l, b, dt = parameters

    def f(x, t, u):
        tick = jnp.array(t/dt, int)
        return jnp.array([
            x[1],
            -b / (m * l ** 2) * x[1] - g / l * jnp.sin(x[0]) + 1 / (m * l ** 2) * u[tick]
        ]) + noise[tick]

    return lambda x, t, u: f(x, t, u)
