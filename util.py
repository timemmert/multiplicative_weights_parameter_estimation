import jax
import jax.numpy as jnp

from constants import dim_state


def get_noise(cov: jnp.ndarray, key_measurement_noise, ticks_end):
    if jnp.linalg.norm(cov) == 0:
        return jnp.zeros((ticks_end + 1, dim_state,))
    return jax.random.multivariate_normal(
        key=key_measurement_noise, mean=jnp.zeros((dim_state,)), cov=cov, shape=(ticks_end + 1,))
