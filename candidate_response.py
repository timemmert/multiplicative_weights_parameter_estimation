import jax
from jax import vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from constants import dim_state, dt


def simulate_controlled_system(K, f, ts, x_goal, x_start, perturbation=None):
    if perturbation is None:
        perturbation = jnp.zeros((len(ts) - 1,))
    x = jnp.zeros((len(ts), dim_state))
    x = x.at[0].set(x_start)
    u = jnp.zeros((len(ts) - 1, 1,))
    for horizon_tick, t in enumerate(ts[:-1]):
        e = x_goal - x[horizon_tick]
        u = u.at[horizon_tick].set(K @ e)

        x = x.at[horizon_tick + 1].set(odeint(
            f,
            x[horizon_tick],
            jnp.array([t, t + dt]),
            u[horizon_tick],
        )[-1] + perturbation[horizon_tick])
    return x, u


def control_pieces(f_i, ts, u, x_measure):
    odeint_vec = vmap(lambda x_start_, ts_, u_: odeint(f_i, x_start_, ts_, u_, ))
    ts_expanded = jnp.expand_dims(ts, axis=1)
    t_tuples = jnp.concatenate((ts_expanded, dt + ts_expanded,), axis=1)[:-1]
    x_i = jnp.concatenate(
        (
            jnp.expand_dims(x_measure[0], axis=0),
            odeint_vec(x_measure[:-1], t_tuples, u)[:, -1, :],
        )
    )
    return x_i
