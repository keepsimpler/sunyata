import jax
import jax.numpy as jnp
from jax import random

from sunyata.equinox.utils import compute_cross_entropy
from sunyata.equinox.utils import one_hot_encode, setup_colab_tpu_or_emulate_it_by_cpus


def test_compute_cross_entropy():
    batch_size, sequence_len, dim_categorical_probabilities = 2, 16, 32
    predicted_categorical_probabilities = random.normal(random.PRNGKey(1), (batch_size, sequence_len, dim_categorical_probabilities))
    predicted_categorical_probabilities = jax.nn.softmax(predicted_categorical_probabilities, axis=-1)
    target_categorical_probabilities = random.normal(random.PRNGKey(1), (batch_size, sequence_len, dim_categorical_probabilities))
    target_categorical_probabilities = jax.nn.softmax(target_categorical_probabilities, axis=-1)

    cross_entropy = compute_cross_entropy(predicted_categorical_probabilities, target_categorical_probabilities)

    assert cross_entropy.shape == (batch_size, sequence_len)
    
    assert jnp.all(cross_entropy > 0)


def test_setup_colab_tpu_or_emulate_it_by_cpus():
    setup_colab_tpu_or_emulate_it_by_cpus()

def test_one_hot_encode():
    batch_size = 2
    sequence_len = 16
    dim_categorical_probabilities = 32

    labels = jax.random.randint(jax.random.PRNGKey(1), (batch_size, sequence_len), minval=0, maxval=dim_categorical_probabilities-1)

    categorical_probabilities = one_hot_encode(labels, dim_categorical_probabilities)

    assert jnp.allclose(jnp.sum(categorical_probabilities, axis=-1), 1.)
