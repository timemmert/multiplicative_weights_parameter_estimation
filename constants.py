import jax
import jax.numpy as jnp
import numpy as np

from losses import loss_maximum_likelihood

key_measurement_noise = jax.random.PRNGKey(1)
dim_state = 2

dt = 0.005

T_end = 3
ticks_end = int(T_end / dt)

horizon_length = 0.01  # shorter horizon -> better performance but also more computational effort. Set to dt for no horizon
horizon_length_ticks = int(horizon_length / dt)

n_horizons = int(np.ceil(T_end / horizon_length))

measurement_noise_on = True
noise_multiplier = 0.00001 if measurement_noise_on else 0
cov_measurement = jnp.eye(dim_state) * noise_multiplier


# Using ground truth control could be hard to realize in the real world since it needs to be transmitted to the estimator
use_ground_truth_control = False
use_individual_control = False
use_pieces_control = True

assert use_ground_truth_control + use_individual_control + use_pieces_control == 1

loss = loss_maximum_likelihood

Q_position = 100
