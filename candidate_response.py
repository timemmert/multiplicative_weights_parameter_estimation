from jax import jit, vmap
from jax._src.lax.control_flow import scan
from jax.experimental.ode import odeint
import jax.numpy as jnp


def simulate_control_system_function(f):
    def step(carry, inputs):
        x, K_, x_goal_, dt_, perturbation_ = carry
        horizon_tick, t = inputs

        e = x_goal_ - x
        u = K_ @ e

        x_next = odeint(f, x, jnp.array([t, t + dt_]), u)[-1] + perturbation_[horizon_tick]

        return (x_next, K_, x_goal_, dt_, perturbation_), (x_next, u)

    def run(K, ts, x_goal, x_start, dt, perturbation):
        initial_carry = (x_start, K, x_goal, dt, perturbation)
        _, (x, u) = scan(step, initial_carry, (jnp.arange(len(ts) - 1), ts[:-1]))
        x = jnp.vstack((x_start[None, :], x))
        return x, u

    return jit(run)


def candidate_control_function(f_i):
    def control_pieces(ts_, u_, x_measure_, dt_):
        odeint_vec = vmap(lambda x_start_, ts_, u_: odeint(f_i, x_start_, ts_, u_, ))
        ts_expanded = jnp.expand_dims(ts_, axis=1)
        t_tuples = jnp.concatenate((ts_expanded, dt_ + ts_expanded,), axis=1)[:-1]
        x_i = jnp.concatenate(
            (
                jnp.expand_dims(x_measure_[0], axis=0),
                odeint_vec(x_measure_[:-1], t_tuples, u_)[:, -1, :],
            )
        )
        return x_i

    return jit(control_pieces)
