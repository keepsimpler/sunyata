import jax.numpy as jnp

def compute_cross_entropy(predicted_categorical_probabilities: jnp.ndarray, target_categorical_probabilities: jnp.ndarray):
    """Compute the cross entropy between predicated categorical distribution and the true categorical distribution"""
    return - jnp.sum(jnp.log(predicted_categorical_probabilities) * target_categorical_probabilities, axis=-1)