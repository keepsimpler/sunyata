import jax
import jax.numpy as jnp

from sunyata.utils import one_hot_encode, setup_colab_tpu_or_emulate_it_by_cpus

def test_setup_colab_tpu_or_emulate_it_by_cpus():
    setup_colab_tpu_or_emulate_it_by_cpus()

def test_one_hot_encode():
    batch_size = 2
    sequence_len = 16
    dim_categorical_probabilities = 32

    labels = jax.random.randint(jax.random.PRNGKey(1), (batch_size, sequence_len), minval=0, maxval=dim_categorical_probabilities-1)

    categorical_probabilities = one_hot_encode(labels, dim_categorical_probabilities)

    assert jnp.allclose(jnp.sum(categorical_probabilities, axis=-1), 1.)
