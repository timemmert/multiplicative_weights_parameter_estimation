import jax.numpy as jnp

from losses import loss_maximum_likelihood, loss_squared

dim_state = 2

x_0 = jnp.array([2.5, 0.])

plus_goal = jnp.array([jnp.pi + 0.3, 0])
minus_goal = jnp.array([jnp.pi - 0.3, 0])

linearize_around = jnp.array([jnp.pi, 0.])

perturbation_on = False
perturbation_multiplier = 0.000001 if perturbation_on else 0

loss = loss_squared

Q_position = 100

print_current_horizon = False

use_original_sim_start = True