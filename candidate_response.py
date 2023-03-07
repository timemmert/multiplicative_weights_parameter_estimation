from jax import vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from constants import dim_state, dt, horizon_length_ticks


def control_using_candidate_error(f_i, K_j, ts, x_goal, x_measure):
    x_i = jnp.concatenate(
        (
            jnp.expand_dims(x_measure[0], axis=0),
            jnp.zeros((horizon_length_ticks, dim_state,)),
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
    return x_i


def control_pieces(f_i, ts, u, x_measure):
    odeint_vec = vmap(lambda x_start_, ts_, u_: odeint(f_i, x_start_, ts_, u_, ))
    u_map = jnp.expand_dims(u, axis=1)
    ts_expanded = jnp.expand_dims(ts, axis=1)
    t_tuples = jnp.concatenate((ts_expanded, dt + ts_expanded,), axis=1)[:-1]
    x_i = jnp.concatenate(
        (
            jnp.expand_dims(x_measure[0], axis=0),
            odeint_vec(x_measure[:-1], t_tuples, u_map)[:, -1, :],
        )
    )
    return x_i
