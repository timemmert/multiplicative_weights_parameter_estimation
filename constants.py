import jax
import jax.numpy as jnp
import numpy as np

from losses import loss_maximum_likelihood, loss_squared

key_perturbation = jax.random.PRNGKey(4)
dim_state = 2

x_0 = jnp.array([2.5, 0.])

plus_goal = jnp.array([jnp.pi + 0.3, 0])
minus_goal = jnp.array([jnp.pi - 0.3, 0])

linearize_around = jnp.array([jnp.pi, 0.])

dt = 0.01
T_end = 10
horizon_length = 0.1   # shorter horizon -> better performance but also more computational effort. Set to dt for no horizon

ticks_end = int(T_end / dt)
horizon_length_ticks = int(horizon_length / dt)
n_horizons = int(np.ceil(T_end / horizon_length))

perturbation_on = False
perturbation_multiplier = 0.000001 if perturbation_on else 0
cov_perturbation = jnp.eye(dim_state) * perturbation_multiplier

loss = loss_maximum_likelihood

Q_position = 100


perturbation = jax.random.multivariate_normal(
    key=key_perturbation,
    mean=jnp.zeros((dim_state,)),
    cov=cov_perturbation,
    shape=(ticks_end,)
) if perturbation_on else jnp.zeros((ticks_end, dim_state,))