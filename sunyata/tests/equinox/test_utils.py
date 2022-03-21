import jax
import jax.numpy as jnp
from jax import random

from sunyata.equinox.utils import compute_cross_entropy

def test_compute_cross_entropy():
    batch_size, sequence_len, dim_categorical_probabilities = 2, 16, 32
    predicted_categorical_probabilities = random.normal(random.PRNGKey(1), (batch_size, sequence_len, dim_categorical_probabilities))
    predicted_categorical_probabilities = jax.nn.softmax(predicted_categorical_probabilities, axis=-1)
    target_categorical_probabilities = random.normal(random.PRNGKey(1), (batch_size, sequence_len, dim_categorical_probabilities))
    target_categorical_probabilities = jax.nn.softmax(target_categorical_probabilities, axis=-1)

    cross_entropy = compute_cross_entropy(predicted_categorical_probabilities, target_categorical_probabilities)

    assert cross_entropy.shape == (batch_size, sequence_len)
    
    assert jnp.all(cross_entropy > 0)


